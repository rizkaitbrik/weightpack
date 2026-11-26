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

	"github.com/opencontainers/go-digest"
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
	ft, err := b.checkFineTune(step, src, parent, base, byID, fail)
	if err != nil {
		return err
	}
	req := &workerpb.EncodeLayerRequest{Codec: workerpb.Codec_CODEC_XOR, TargetFiles: ft.files}

	// The cache remembers the deltas these exact files gave over this exact
	// parent chain; a hit needs no worker at all.
	xorKey := b.cache.xorKey(parent.Chain, req.TargetFiles)
	if hit, ok := b.cache.getXOR(b.store, xorKey); ok {
		for _, e := range hit.Tensors {
			b.blobs[layer.ID] = append(b.blobs[layer.ID], ocispec.Descriptor{
				MediaType: xorMediaType, Digest: digest.Digest(e.Digest), Size: e.Size,
				Annotations: map[string]string{spec.AnnotationLayer: layer.ID, spec.AnnotationTensor: e.Tensor},
			})
		}
		return b.finishXOR(step, layer, parent, len(hit.Tensors), hit.Unchanged, hit.Raw, true)
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
	if req.Base, req.Parents, err = b.parentBlobs(ft, inDir); err != nil {
		return err
	}

	b.progress(api.Event{Step: step, Message: fmt.Sprintf("encoding %s as XOR deltas over %s", src.Source, parent.ID)})
	resp, err := b.worker.EncodeLayer(b.ctx, req)
	if err != nil {
		return fail("%s", WorkerError(err))
	}

	var raw int64
	var entry cachedXOR
	entry.Unchanged = resp.Unchanged
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
		entry.Tensors = append(entry.Tensors, cachedTensor{Tensor: t.Name, Digest: d.Digest.String(), Size: d.Size})
	}
	entry.Raw = raw
	if entry.Tensors == nil {
		entry.Tensors = []cachedTensor{}
	}
	b.cache.store(xorKey, entry)
	return b.finishXOR(step, layer, parent, len(resp.Tensors), resp.Unchanged, raw, false)
}

// fineTune is a full fine-tune to import as a delta over its parent: its
// weight files, checked against the base, and the chain below it, which is
// the base and XOR layers only.
type fineTune struct {
	dir        string
	files      []string
	baseLayer  *spec.Layer
	xorParents []*spec.Layer // bottom first
}

// checkFineTune checks a fine-tune before any work: delta layers sit only on
// the base or on XOR layers, whose bits are exact (over a LoRA layer the
// parent would come from float math, whose rounding can differ between
// machines); the fine-tune has exactly the base's tensors; and the parent's
// tokenizer and config, which the image keeps, fit it.
func (b *builder) checkFineTune(step string, src weightfile.Layer, parent *spec.Layer, base *baseModel, byID map[string]*spec.Layer, fail func(string, ...any) error) (*fineTune, error) {
	ft := &fineTune{baseLayer: parent}
	for ; ft.baseLayer.Kind != spec.KindBase; ft.baseLayer = byID[ft.baseLayer.Parent] {
		if ft.baseLayer.Kind != spec.KindDeltaXOR {
			return nil, fail("a codec=%s layer sits only on the base or on XOR layers, but %s is a %s layer: "+
				"its weights come from float math, so the delta would not be exact; put delta layers below LoRA layers", src.Codec, ft.baseLayer.ID, ft.baseLayer.Kind)
		}
		ft.xorParents = append([]*spec.Layer{ft.baseLayer}, ft.xorParents...)
	}
	dir, err := b.sourceDir(step, src.Line, src.Source, fetchModel)
	if err != nil {
		return nil, err
	}
	ft.dir = dir
	names, err := weightFiles(dir)
	if err != nil {
		return nil, fail("%v", err)
	}
	seen := map[string]bool{}
	for _, n := range names {
		p := filepath.Join(dir, n)
		f, err := safetensors.Open(p)
		if err != nil {
			return nil, fail("%v", err)
		}
		for _, t := range f.Tensors {
			want, ok := base.tensors[t.Name]
			if !ok || want.DType != t.DType || !slices.Equal(want.Shape, t.Shape) {
				_ = f.Close()
				return nil, fail("%s is %s%v in %s, which the base does not have; is this a fine-tune of another model?", t.Name, t.DType, t.Shape, n)
			}
			seen[t.Name] = true
		}
		_ = f.Close()
		ft.files = append(ft.files, p)
	}
	if len(seen) != len(base.tensors) {
		return nil, fail("the fine-tune has %d of the base's %d tensors; it must have them all", len(seen), len(base.tensors))
	}
	same, supplied, err := b.sameTokenizer(dir, base.tokenizer)
	switch {
	case err != nil:
		return nil, fail("%v", err)
	case !supplied:
		b.progress(api.Event{Step: step, Warning: true, Message: "the fine-tune has no tokenizer of its own; the image keeps its parent's"})
	case !same:
		return nil, fail("the fine-tune's tokenizer differs from its parent's, so its token ids mean other things; " +
			"an import keeps the parent's tokenizer, so this fine-tune cannot be imported over it")
	}
	differs, supplied, err := b.configDiff(dir, b.blobs[ft.baseLayer.ID])
	switch {
	case err != nil:
		return nil, fail("%v", err)
	case !supplied:
		b.progress(api.Event{Step: step, Warning: true, Message: "the fine-tune has no config.json of its own; the image keeps its parent's"})
	case len(differs) > 0:
		return nil, fail("the fine-tune's config.json differs from its parent's in %s; an import keeps the parent's config, "+
			"so the model would run with the parent's settings: build the fine-tune as a base of its own (FROM) instead", strings.Join(differs, ", "))
	}
	return ft, nil
}

// parentBlobs are what the worker needs to rebuild a fine-tune's parent:
// the base's tensor blobs, and the XOR layers above it decoded into inDir.
func (b *builder) parentBlobs(ft *fineTune, inDir string) ([]*workerpb.Blob, []*workerpb.XORLayer, error) {
	var base []*workerpb.Blob
	for _, d := range b.blobs[ft.baseLayer.ID] {
		if spec.BaseMediaType(d.MediaType) != spec.MediaTypeTensor {
			continue
		}
		p, err := b.store.Path(d.Digest.String())
		if err != nil {
			return nil, nil, err
		}
		base = append(base, &workerpb.Blob{Digest: d.Digest.String(), MediaType: d.MediaType, Path: p})
	}
	var parents []*workerpb.XORLayer
	for _, x := range ft.xorParents {
		layerReq, err := DecodeXORLayer(b.store, x.ID, b.blobs[x.ID], inDir)
		if err != nil {
			return nil, nil, err
		}
		parents = append(parents, layerReq)
	}
	return base, parents, nil
}

// finishXOR seals an XOR layer once its blobs are in place.
func (b *builder) finishXOR(step string, layer, parent *spec.Layer, changed int, unchanged, raw int64, cached bool) error {
	var stored int64
	for _, d := range b.blobs[layer.ID] {
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
	note := ""
	if cached {
		note = " (from the build cache)"
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("XOR delta: %d tensors changed, %d unchanged; %s%s",
		changed, unchanged, ratio, note)})
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

// configMetadata are config.json keys that record how a checkpoint was saved,
// not how the model computes; a fine-tune may differ from its base in them.
var configMetadata = []string{
	"_name_or_path", "transformers_version", "torch_dtype", "dtype", "use_cache",
	"_attn_implementation", "_attn_implementation_autoset", "attn_implementation",
}

// configDiff compares the fine-tune's config.json in dir with the base's
// config.json blob among baseBlobs, and returns the keys whose values
// differ, metadata aside. supplied is false when dir has no config.json.
func (b *builder) configDiff(dir string, baseBlobs []ocispec.Descriptor) (differs []string, supplied bool, err error) {
	mine, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if os.IsNotExist(err) {
		return nil, false, nil
	}
	if err != nil {
		return nil, true, err
	}
	var parentDigest string
	for _, d := range baseBlobs {
		if spec.BaseMediaType(d.MediaType) == spec.MediaTypeMeta && d.Annotations[spec.AnnotationTitle] == "config.json" {
			parentDigest = d.Digest.String()
		}
	}
	if parentDigest == "" {
		return nil, true, fmt.Errorf("the base has no config.json to compare the fine-tune's with")
	}
	parent, err := b.store.ReadBlob(parentDigest)
	if err != nil {
		return nil, true, err
	}
	var a, p map[string]any
	if err := json.Unmarshal(mine, &a); err != nil {
		return nil, true, fmt.Errorf("the fine-tune's config.json: %w", err)
	}
	if err := json.Unmarshal(parent, &p); err != nil {
		return nil, true, fmt.Errorf("the base's config.json: %w", err)
	}
	for _, k := range configMetadata {
		delete(a, k)
		delete(p, k)
	}
	keys := map[string]bool{}
	for k := range a {
		keys[k] = true
	}
	for k := range p {
		keys[k] = true
	}
	for k := range keys {
		if !reflect.DeepEqual(a[k], p[k]) {
			differs = append(differs, k)
		}
	}
	slices.Sort(differs)
	return differs, true, nil
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
// safetensors, for a worker to read, and describes them.
func DecodeXORLayer(s *store.Store, id string, blobs []ocispec.Descriptor, dir string) (*workerpb.XORLayer, error) {
	var xor []ocispec.Descriptor
	for _, d := range blobs {
		if spec.BaseMediaType(d.MediaType) == spec.MediaTypeDeltaXOR {
			xor = append(xor, d)
		}
	}
	out, err := DecodeBlobs(s, xor, dir)
	if err != nil {
		return nil, err
	}
	return &workerpb.XORLayer{Id: id, Blobs: out}, nil
}

// DecodeBlobs gives a worker stored blobs as plain safetensors: a blob
// stored plain as its own path in the store, and a compressed one as a file
// decoded into dir. The files get random names made inside dir: a layer id
// comes from an image's config, which a pulled image controls, so it never
// takes part in a path.
func DecodeBlobs(s *store.Store, blobs []ocispec.Descriptor, dir string) ([]*workerpb.Blob, error) {
	var out []*workerpb.Blob
	for _, d := range blobs {
		if entropy.Suffix(d.MediaType) == "" {
			p, err := s.Path(d.Digest.String())
			if err != nil {
				return nil, err
			}
			out = append(out, &workerpb.Blob{Digest: d.Digest.String(), MediaType: d.MediaType, Path: p})
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
		f, err := os.CreateTemp(dir, "blob-*.safetensors")
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
			return nil, fmt.Errorf("decode blob %s: %w", d.Digest, err)
		}
		out = append(out, &workerpb.Blob{Digest: d.Digest.String(), MediaType: d.MediaType, Path: p})
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
