package build

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"go.yaml.in/yaml/v3"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// Lossy delta entries, as weightpack_worker/lossy.py writes them: a
// tensor's delta file holds these, named <tensor><suffix>.
const (
	lossySigns   = ".signs"   // bit1: U8, the sign bits packed eight to a byte
	lossyScale   = ".scale"   // bit1: F32[1]
	lossyIndices = ".indices" // sparse: U32 or I64, flat indices, ascending
	lossyValues  = ".values"  // sparse: F32
	lossyDelta   = ".delta"   // a tensor kept whole: F32, the tensor's shape
)

var lossyCodecs = map[string]struct {
	codec     workerpb.Codec
	mediaType string
}{
	weightfile.CodecBit1:   {workerpb.Codec_CODEC_BIT1, spec.MediaTypeDeltaBit1},
	weightfile.CodecSparse: {workerpb.Codec_CODEC_SPARSE, spec.MediaTypeDeltaSparse},
}

// cachedLossy is what a lossy import of these files over this chain gave:
// its blobs and its eval, so a rebuild needs no worker at all.
type cachedLossy struct {
	Tensors       []cachedTensor `json:"tensors"`
	Unchanged     int64          `json:"unchanged"`
	Raw           int64          `json:"raw"` // the changed tensors' own size
	RelativeError float64        `json:"relative_error"`
	SourceDigest  string         `json:"source_digest"`
	Suite         string         `json:"suite"`
	Examples      int32          `json:"examples"`
	Tokens        int64          `json:"tokens"`
	SourceScore   float64        `json:"source_score"`
	Score         float64        `json:"score"`
}

// ingestLossy imports a full fine-tune as a lossy delta over its parent
// (codec=bit1 or sparse), gated on an eval. After the checks an XOR import
// makes, the worker encodes the deltas and writes the model they give back;
// it then scores the fine-tune and that model on the suite (eval=, or the
// fine-tune's own replies to general prompts). The build fails when the
// drop passes max_drop, and otherwise records it in the layer's fidelity
// record with the scores, the suite and the source's digest.
func (b *builder) ingestLossy(step string, src weightfile.Layer, layer, parent *spec.Layer, base *baseModel, byID map[string]*spec.Layer) error {
	fail := func(format string, args ...any) error {
		return &weightfile.Error{File: b.file, Line: src.Line, Msg: step + ": " + fmt.Sprintf(format, args...)}
	}
	codec := lossyCodecs[src.Codec]
	if b.worker == nil {
		return fail("codec=%s runs in the build worker; start wpkd with --worker", src.Codec)
	}
	ft, err := b.checkFineTune(step, src, parent, base, byID, fail)
	if err != nil {
		return err
	}
	density := 0.0
	if src.Codec == weightfile.CodecSparse {
		density = weightfile.DefaultDensity
		if src.Density != nil {
			density = *src.Density
		}
	}
	suite, suiteDigest := "", ""
	if src.Eval != "" {
		suite = b.resolve(src.Eval)
		if suiteDigest, err = suiteFilesDigest(suite); err != nil {
			return fail("eval: %v", err)
		}
	}
	template, err := b.chatTemplate(b.wf)
	if err != nil {
		return fail("%v", err)
	}
	info, err := b.worker.Info(b.ctx, &workerpb.InfoRequest{})
	if err != nil {
		return fail("ask the build worker its version: %s", WorkerError(err))
	}

	key := ""
	if fk := b.cache.xorKey(parent.Chain, ft.files); fk != "" {
		key = hashKey("lossy", src.Codec, strconv.FormatFloat(density, 'g', -1, 64), fk, suiteDigest, template.digest, info.Version)
	}
	var entry cachedLossy
	cached := b.cache.load(key, &entry) && entry.Tensors != nil && b.storeHas(entry.Tensors)
	if cached {
		b.cache.hits += len(entry.Tensors)
	} else {
		if entry, err = b.encodeLossy(step, src, layer, parent, base, ft, codec.codec, codec.mediaType, density, suite, template, fail); err != nil {
			return err
		}
		b.cache.store(key, entry)
	}
	b.blobs[layer.ID] = nil
	for _, t := range entry.Tensors {
		b.blobs[layer.ID] = append(b.blobs[layer.ID], ocispec.Descriptor{
			MediaType: codec.mediaType, Digest: digest.Digest(t.Digest), Size: t.Size,
			Annotations: map[string]string{spec.AnnotationLayer: layer.ID, spec.AnnotationTensor: t.Tensor},
		})
	}

	var stored int64
	var refs []spec.BlobRef
	for _, d := range b.blobs[layer.ID] {
		stored += d.Size
		refs = append(refs, spec.BlobRefFromDescriptor(d))
	}
	note := ""
	if cached {
		note = " (from the build cache)"
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("%s delta: %d tensors changed, %d unchanged; %s of changed weights stored in %s (%.1f%%), relative error %.3f%s",
		src.Codec, len(entry.Tensors), entry.Unchanged, api.FormatSize(entry.Raw), api.FormatSize(stored), 100*float64(stored)/math.Max(1, float64(entry.Raw)), entry.RelativeError, note)})

	drop := entry.SourceScore - entry.Score
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("eval %s (%d examples, %d tokens): the fine-tune predicts %.2f%% of tokens, the %s layer %.2f%%: a drop of %.2f points (max_drop %g)",
		entry.Suite, entry.Examples, entry.Tokens, entry.SourceScore, src.Codec, entry.Score, drop, *src.MaxDrop)})
	if drop > *src.MaxDrop {
		hint := "keep more of the delta (codec=sparse with a higher density=, or codec=xor, which is exact)"
		return fail("the %s layer's drop on %s is %.2f points, more than max_drop=%g; %s, or raise max_drop", src.Codec, entry.Suite, drop, *src.MaxDrop, hint)
	}

	fid := layer.Fidelity
	if fid == nil {
		fid = &spec.Fidelity{Source: src.Source.String(), Codec: src.Codec, MaxDrop: *src.MaxDrop}
		layer.Fidelity = fid
	}
	fid.SourceDigest = entry.SourceDigest
	fid.Density = density
	fid.Eval, fid.EvalDigest = entry.Suite, suiteDigest
	fid.SourceScore, fid.Score, fid.MeasuredDrop = &entry.SourceScore, &entry.Score, &drop
	layer.Digest = spec.LayerDigest(*layer, refs, nil)
	layer.Chain = spec.ChainID(parent.Chain, layer.Digest)
	return nil
}

// encodeLossy has the worker encode the deltas and evaluate them, then
// checks and stores the deltas.
func (b *builder) encodeLossy(step string, src weightfile.Layer, layer, parent *spec.Layer, base *baseModel, ft *fineTune, codec workerpb.Codec, mediaType string, density float64, suite string, template templateSource, fail func(string, ...any) error) (cachedLossy, error) {
	var entry cachedLossy
	staging, err := b.store.TempDir("encode-")
	if err != nil {
		return entry, err
	}
	defer os.RemoveAll(staging)
	inDir, outDir, recon := filepath.Join(staging, "in"), filepath.Join(staging, "out"), filepath.Join(staging, "model")
	for _, d := range []string{inDir, outDir, recon} {
		if err := os.Mkdir(d, 0o700); err != nil {
			return entry, err
		}
	}
	req := &workerpb.EncodeLayerRequest{Codec: codec, TargetFiles: ft.files, OutDir: outDir, Density: density, ReconstructDir: recon}
	if req.Base, req.Parents, err = b.parentBlobs(ft, inDir); err != nil {
		return entry, err
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("encoding %s as a %s delta over %s", src.Source, src.Codec, parent.ID)})
	resp, err := b.worker.EncodeLayer(b.ctx, req)
	if err != nil {
		return entry, fail("%s", WorkerError(err))
	}
	entry.Unchanged, entry.RelativeError = resp.Unchanged, resp.RelativeError
	for _, t := range resp.Tensors {
		want, ok := base.tensors[t.Name]
		if !ok {
			return entry, fail("worker returned tensor %q, which the base does not have", t.Name)
		}
		if t.File == "" || t.File != filepath.Base(t.File) || strings.HasPrefix(t.File, ".") {
			return entry, fail("worker returned file %q outside its output directory", t.File)
		}
		d, err := b.putLossy(filepath.Join(outDir, t.File), t.Name, want, src.Codec, mediaType, layer.ID)
		if err != nil {
			return entry, fail("worker output for %s: %v", t.Name, err)
		}
		size, _ := safetensors.ByteSize(want.DType, want.Shape)
		entry.Raw += size
		entry.Tensors = append(entry.Tensors, cachedTensor{Tensor: t.Name, Digest: d.Digest.String(), Size: d.Size})
	}
	if entry.Tensors == nil {
		entry.Tensors = []cachedTensor{}
	}
	if entry.SourceDigest, err = filesDigest(ft.files); err != nil {
		return entry, err
	}

	what := "its own replies to general prompts"
	if suite != "" {
		what = src.Eval
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("evaluating the fine-tune and the %s layer on %s", src.Codec, what)})
	templatePath, err := template.write(b, staging)
	if err != nil {
		return entry, err
	}
	ev, err := b.worker.Evaluate(b.ctx, &workerpb.EvaluateRequest{Models: []string{ft.dir, recon}, Suite: suite, ChatTemplate: templatePath})
	if err != nil {
		return entry, fail("eval: %s", WorkerError(err))
	}
	if len(ev.Scores) != 2 {
		return entry, fail("eval: the worker scored %d models, want 2", len(ev.Scores))
	}
	entry.Suite, entry.Examples, entry.Tokens = ev.Suite, ev.Examples, ev.Tokens
	entry.SourceScore, entry.Score = ev.Scores[0].Accuracy, ev.Scores[1].Accuracy
	return entry, nil
}

// putLossy checks a worker's delta file for one tensor against the codec's
// layout and the base tensor, and stores it as a canonical blob.
func (b *builder) putLossy(path, name string, want safetensors.Tensor, codec, mediaType, layerID string) (ocispec.Descriptor, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	defer f.Close()
	if !isFloat(want.DType) {
		return ocispec.Descriptor{}, fmt.Errorf("the base tensor is %s; lossy codecs take floats only", want.DType)
	}
	n := int64(1)
	for _, d := range want.Shape {
		n *= d
	}
	type entry struct {
		dtypes []string
		shape  func([]int64) bool
	}
	expect := map[string]entry{}
	switch {
	case len(want.Shape) < 2:
		expect[lossyDelta] = entry{[]string{"F32"}, func(s []int64) bool { return equalShape(s, want.Shape) }}
	case codec == weightfile.CodecBit1:
		expect[lossySigns] = entry{[]string{"U8"}, func(s []int64) bool { return equalShape(s, []int64{(n + 7) / 8}) }}
		expect[lossyScale] = entry{[]string{"F32"}, func(s []int64) bool { return equalShape(s, []int64{1}) }}
	default:
		var k int64 = -1
		for _, t := range f.Tensors {
			if t.Name == name+lossyValues && len(t.Shape) == 1 {
				k = t.Shape[0]
			}
		}
		count := func(s []int64) bool { return len(s) == 1 && s[0] == k && k >= 1 && k <= n }
		expect[lossyIndices] = entry{[]string{"U32", "I64"}, count}
		expect[lossyValues] = entry{[]string{"F32"}, count}
	}
	if len(f.Tensors) != len(expect) {
		return ocispec.Descriptor{}, fmt.Errorf("%d tensors, want %d", len(f.Tensors), len(expect))
	}
	var refs []tensorRef
	for _, t := range f.Tensors {
		e, ok := expect[strings.TrimPrefix(t.Name, name)]
		if !ok || !strings.HasPrefix(t.Name, name) || !containsString(e.dtypes, t.DType) || !e.shape(t.Shape) {
			return ocispec.Descriptor{}, fmt.Errorf("%s is %s%v, which a %s delta of %s%v does not hold", t.Name, t.DType, t.Shape, codec, want.DType, want.Shape)
		}
		refs = append(refs, tensorRef{file: f, t: t, name: t.Name})
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].name < refs[j].name })
	return b.putTensors(refs, mediaType, map[string]string{spec.AnnotationLayer: layerID, spec.AnnotationTensor: name})
}

func isFloat(dtype string) bool {
	return dtype == "F16" || dtype == "BF16" || dtype == "F32" || dtype == "F64"
}

// storeHas reports whether the store still has every cached blob.
func (b *builder) storeHas(ts []cachedTensor) bool {
	for _, t := range ts {
		if ok, err := b.store.Has(t.Digest); err != nil || !ok {
			return false
		}
	}
	return true
}

// filesDigest is sha256 over the files' own sha256 digests, in name order.
func filesDigest(files []string) (string, error) {
	sorted := append([]string{}, files...)
	sort.Slice(sorted, func(i, j int) bool { return filepath.Base(sorted[i]) < filepath.Base(sorted[j]) })
	h := sha256.New()
	for _, f := range sorted {
		d, err := fileDigest(f)
		if err != nil {
			return "", err
		}
		h.Write([]byte(strings.TrimPrefix(d, "sha256:")))
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

// suiteFilesDigest covers an eval suite's files: the suite file, and the
// data file a YAML suite names.
func suiteFilesDigest(path string) (string, error) {
	files := []string{path}
	if ext := filepath.Ext(path); ext == ".yaml" || ext == ".yml" {
		raw, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		var s struct {
			Data string `yaml:"data"`
		}
		if err := yaml.Unmarshal(raw, &s); err != nil || s.Data == "" {
			return "", fmt.Errorf("%s: want a mapping with data: <examples.jsonl>", path)
		}
		files = append(files, filepath.Join(filepath.Dir(path), s.Data))
	}
	h := sha256.New()
	for _, f := range files {
		d, err := fileDigest(f)
		if err != nil {
			return "", err
		}
		h.Write([]byte(d))
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}
