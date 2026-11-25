package build

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/entropy"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// xorMediaType is how XOR deltas are stored: delta.xor.v1, zstd-compressed.
const xorMediaType = spec.MediaTypeDeltaXOR + "+" + entropy.Zstd

// ingestXOR imports a full fine-tune as a lossless XOR delta over its
// parent. The worker computes the deltas; this checks each one, writes it in
// canonical form and compresses it.
//
// XOR layers sit only on the base or on other XOR layers: their parent's
// bits are then exact. Over a LoRA layer the parent would be the result of
// float math, whose rounding can differ between machines, and decoding would
// no longer give the fine-tune back bit for bit.
func (b *builder) ingestXOR(step string, src weightfile.Layer, layer, parent *spec.Layer, base *baseModel, byID map[string]*spec.Layer) error {
	fail := func(format string, args ...any) error {
		return &weightfile.Error{File: b.file, Line: src.Line, Msg: step + ": " + fmt.Sprintf(format, args...)}
	}
	if b.worker == nil {
		return fail("codec=xor runs in the build worker; start wpkd with --worker")
	}
	var xorParents []*spec.Layer
	baseLayer := parent
	for ; baseLayer.Kind != spec.KindBase; baseLayer = byID[baseLayer.Parent] {
		if baseLayer.Kind != spec.KindDeltaXOR {
			return fail("an XOR layer sits only on the base or on other XOR layers, but %s is a %s layer: "+
				"its weights come from float math, so the delta would not be exact; put XOR layers below LoRA layers", baseLayer.ID, baseLayer.Kind)
		}
		xorParents = append([]*spec.Layer{baseLayer}, xorParents...)
	}

	// Check the fine-tune's tensor table against the base before any work.
	dir, err := b.sourceDir(step, src.Line, src.Source, fetchModel)
	if err != nil {
		return err
	}
	names, err := weightFiles(dir)
	if err != nil {
		return fail("%v", err)
	}
	req := &workerpb.EncodeLayerRequest{Codec: workerpb.Codec_CODEC_XOR}
	seen := map[string]bool{}
	for _, n := range names {
		p := filepath.Join(dir, n)
		f, err := safetensors.Open(p)
		if err != nil {
			return fail("%v", err)
		}
		for _, t := range f.Tensors {
			want, ok := base.tensors[t.Name]
			if !ok || want.DType != t.DType || !slices.Equal(want.Shape, t.Shape) {
				_ = f.Close()
				return fail("%s is %s%v in %s, which the base does not have; is this a fine-tune of another model?", t.Name, t.DType, t.Shape, n)
			}
			seen[t.Name] = true
		}
		_ = f.Close()
		req.TargetFiles = append(req.TargetFiles, p)
	}
	if len(seen) != len(base.tensors) {
		return fail("the fine-tune has %d of the base's %d tensors; it must have them all", len(seen), len(base.tensors))
	}
	same, supplied, err := b.sameTokenizer(dir, base.tokenizer)
	switch {
	case err != nil:
		return fail("%v", err)
	case !supplied:
		b.progress(api.Event{Step: step, Warning: true, Message: "the fine-tune has no tokenizer of its own; the image keeps its parent's"})
	case !same:
		return fail("the fine-tune's tokenizer differs from its parent's, so its token ids mean other things; " +
			"a lossless import keeps the parent's tokenizer, so this fine-tune cannot be imported over it")
	}

	staging, err := b.store.TempDir("encode-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(staging)
	inDir, outDir := filepath.Join(staging, "in"), filepath.Join(staging, "out")
	for _, d := range []string{inDir, outDir} {
		if err := os.Mkdir(d, 0o700); err != nil {
			return err
		}
	}
	req.OutDir = outDir
	for _, d := range b.blobs[baseLayer.ID] {
		if spec.BaseMediaType(d.MediaType) != spec.MediaTypeTensor {
			continue
		}
		p, err := b.store.Path(d.Digest.String())
		if err != nil {
			return err
		}
		req.Base = append(req.Base, &workerpb.Blob{Digest: d.Digest.String(), MediaType: d.MediaType, Path: p})
	}
	for _, x := range xorParents {
		layerReq, err := DecodeXORLayer(b.store, x.ID, b.blobs[x.ID], inDir)
		if err != nil {
			return err
		}
		req.Parents = append(req.Parents, layerReq)
	}

	b.progress(api.Event{Step: step, Message: fmt.Sprintf("encoding %s as XOR deltas over %s", src.Source, parent.ID)})
	resp, err := b.worker.EncodeLayer(b.ctx, req)
	if err != nil {
		return fail("%s", WorkerError(err))
	}

	var raw, stored int64
	for _, t := range resp.Tensors {
		want := base.tensors[t.Name]
		if t.File == "" || t.File != filepath.Base(t.File) || strings.HasPrefix(t.File, ".") {
			return fail("worker returned file %q outside its output directory", t.File)
		}
		if _, ok := base.tensors[t.Name]; !ok {
			return fail("worker returned tensor %q, which the base does not have", t.Name)
		}
		f, err := safetensors.Open(filepath.Join(outDir, t.File))
		if err != nil {
			return fail("worker output: %v", err)
		}
		got, ok := f.Lookup(t.Name)
		if !ok || len(f.Tensors) != 1 || got.DType != want.DType || !slices.Equal(got.Shape, want.Shape) {
			_ = f.Close()
			return fail("worker output %s does not hold %s as %s%v", t.File, t.Name, want.DType, want.Shape)
		}
		canon, size, err := safetensors.NewReader([]safetensors.Entry{{Name: t.Name, DType: got.DType, Shape: got.Shape, Data: f.Data(got)}}, nil)
		if err != nil {
			_ = f.Close()
			return err
		}
		zr, err := entropy.Encode(entropy.Zstd, canon)
		if err != nil {
			_ = f.Close()
			return err
		}
		d, err := b.put(zr, xorMediaType, map[string]string{spec.AnnotationLayer: layer.ID, spec.AnnotationTensor: t.Name})
		_ = zr.Close() // stops the encoder if put gave up before reading it all
		_ = f.Close()
		if err != nil {
			return fail("%s: %v", t.Name, err)
		}
		b.blobs[layer.ID] = append(b.blobs[layer.ID], d)
		raw += size
		stored += d.Size
	}

	var refs []spec.BlobRef
	for _, d := range b.blobs[layer.ID] {
		refs = append(refs, spec.BlobRefFromDescriptor(d))
	}
	layer.Kind = spec.KindDeltaXOR
	layer.Digest = spec.LayerDigest(*layer, refs, nil)
	layer.Chain = spec.ChainID(parent.Chain, layer.Digest)
	ratio := "nothing changed"
	if raw > 0 {
		ratio = fmt.Sprintf("%s of deltas stored in %s (%.0f%%)", api.FormatSize(raw), api.FormatSize(stored), 100*float64(stored)/float64(raw))
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("XOR delta: %d tensors changed, %d unchanged; %s",
		len(resp.Tensors), resp.Unchanged, ratio)})
	return nil
}

// sameTokenizer compares the tokenizer a fine-tune's directory carries
// (tokenizer.json, else tokenizer.model) with its parent's, whose digest is
// want. supplied is false when the directory has no tokenizer. Byte-equal
// files are the same; two tokenizer.json files are also the same when they
// differ only in formatting, key order or the fields that do not change
// tokenization (version, truncation, padding), since fine-tuning tools often
// re-save the file.
func (b *builder) sameTokenizer(dir, want string) (same, supplied bool, err error) {
	var name string
	for _, n := range []string{"tokenizer.json", "tokenizer.model"} {
		if info, err := os.Stat(filepath.Join(dir, n)); err == nil && info.Mode().IsRegular() {
			name = n
			break
		}
	}
	if name == "" {
		return false, false, nil
	}
	mine, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return false, true, err
	}
	sum := sha256.Sum256(mine)
	if "sha256:"+hex.EncodeToString(sum[:]) == want {
		return true, true, nil
	}
	if name != "tokenizer.json" {
		return false, true, nil
	}
	parent, err := b.store.ReadBlob(want)
	if err != nil {
		return false, true, err
	}
	a, errA := tokenizerSemantics(mine)
	p, errB := tokenizerSemantics(parent)
	if errA != nil || errB != nil {
		return false, true, nil
	}
	return reflect.DeepEqual(a, p), true, nil
}

// tokenizerSemantics parses a tokenizer.json and drops the fields that do
// not change how text maps to token ids.
func tokenizerSemantics(raw []byte) (map[string]any, error) {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	for _, k := range []string{"version", "truncation", "padding"} {
		delete(m, k)
	}
	return m, nil
}

// DecodeXORLayer writes a stored XOR layer's blobs into dir as plain
// safetensors, for a worker to read, and describes them. The files get
// random names made inside dir: the layer id comes from an image's config,
// which a pulled image controls, so it never takes part in a path.
func DecodeXORLayer(s *store.Store, id string, blobs []ocispec.Descriptor, dir string) (*workerpb.XORLayer, error) {
	out := &workerpb.XORLayer{Id: id}
	for _, d := range blobs {
		if spec.BaseMediaType(d.MediaType) != spec.MediaTypeDeltaXOR {
			continue
		}
		rc, err := s.Open(d.Digest.String())
		if err != nil {
			return nil, err
		}
		plain, err := entropy.Decode(d.MediaType, rc)
		if err != nil {
			return nil, err
		}
		f, err := os.CreateTemp(dir, "xor-*.safetensors")
		if err != nil {
			_ = plain.Close()
			return nil, err
		}
		p := f.Name()
		_, err = io.Copy(f, plain)
		_ = plain.Close()
		if cerr := f.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return nil, fmt.Errorf("decode XOR blob %s: %w", d.Digest, err)
		}
		out.Blobs = append(out.Blobs, &workerpb.Blob{Digest: d.Digest.String(), MediaType: d.MediaType, Path: p})
	}
	return out, nil
}

// WorkerError turns a worker call's error into a message: the worker's own
// for a failed call, and what to do when no worker answers.
func WorkerError(err error) string {
	s, ok := status.FromError(err)
	switch {
	case !ok:
		return err.Error()
	case s.Code() == codes.Unavailable:
		return "no build worker answers; start one with weightpack-worker, or start wpkd with --worker"
	default:
		return "worker: " + s.Message()
	}
}
