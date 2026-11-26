package build

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
)

// PackThreshold is the size below which a base tensor shares one packed blob
// with the layer's other small tensors (norms, biases) instead of getting a
// blob of its own.
const PackThreshold = 64 << 10

// baseMetaFiles are the files of a base model directory stored as meta
// blobs. Anything else in the directory is ignored.
var baseMetaFiles = []string{
	"config.json", "generation_config.json",
	"tokenizer.json", "tokenizer_config.json", "tokenizer.model",
	"special_tokens_map.json", "added_tokens.json",
	"vocab.json", "vocab.txt", "merges.txt",
	"chat_template.jinja", "chat_template.json",
	"README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE", "USE_POLICY.md",
}

// pickleExts are weight formats that can run code when loaded. They are
// never read: the trust boundary is safetensors only.
var pickleExts = []string{".bin", ".pt", ".pth", ".ckpt", ".pkl"}

// baseModel is what later layers are checked against.
type baseModel struct {
	tensors   map[string]safetensors.Tensor
	arch      spec.Arch
	tokenizer string
}

type hfConfig struct {
	ModelType string `json:"model_type"`
	Layers    int    `json:"num_hidden_layers"`
	Hidden    int    `json:"hidden_size"`
}

func (b *builder) ingestBase(from weightfile.From, layer *spec.Layer) (*baseModel, error) {
	step := "FROM " + from.Source.String()
	fail := func(format string, args ...any) error {
		return &weightfile.Error{File: b.file, Line: from.Line, Msg: step + ": " + fmt.Sprintf(format, args...)}
	}
	dir, err := b.sourceDir(step, from.Line, from.Source, fetchModel)
	if err != nil {
		return nil, err
	}
	if info, err := os.Stat(dir); err != nil {
		return nil, fail("%v", err)
	} else if !info.IsDir() {
		return nil, fail("%s is not a directory", dir)
	}

	names, err := weightFiles(dir)
	if err != nil {
		return nil, fail("%v", err)
	}
	var files []*safetensors.File
	defer func() {
		for _, f := range files {
			_ = f.Close()
		}
	}()
	type located struct {
		file *safetensors.File
		t    safetensors.Tensor
	}
	var all []located
	owner := map[string]string{}
	var total int64
	for _, name := range names {
		f, err := safetensors.Open(filepath.Join(dir, name))
		if err != nil {
			return nil, fail("%v", err)
		}
		files = append(files, f)
		for _, t := range f.Tensors {
			if prev, dup := owner[t.Name]; dup {
				return nil, fail("tensor %s is in both %s and %s", t.Name, prev, name)
			}
			owner[t.Name] = name
			all = append(all, located{f, t})
			total += t.Size
		}
	}
	if len(all) == 0 {
		return nil, fail("no tensors in %d safetensors files", len(names))
	}
	sort.Slice(all, func(i, j int) bool { return all[i].t.Name < all[j].t.Name })
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("%d tensors in %s, %s", len(all), plural(len(names), "file"), api.FormatSize(total))})

	// Meta files first: the tokenizer digest is part of the fingerprint.
	var tokenizerJSON, tokenizerModel string
	var hasConfig bool
	for _, name := range baseMetaFiles {
		path := filepath.Join(dir, name)
		if info, err := os.Stat(path); err != nil || !info.Mode().IsRegular() {
			continue
		}
		d, err := b.putFile(path, map[string]string{spec.AnnotationLayer: layer.ID, spec.AnnotationTitle: name})
		if err != nil {
			return nil, fail("%v", err)
		}
		b.blobs[layer.ID] = append(b.blobs[layer.ID], d)
		switch name {
		case "tokenizer.json":
			tokenizerJSON = d.Digest.String()
		case "tokenizer.model":
			tokenizerModel = d.Digest.String()
		case "config.json":
			hasConfig = true
		}
	}
	if !hasConfig {
		return nil, fail("no config.json")
	}
	tokenizer := tokenizerJSON
	if tokenizer == "" {
		tokenizer = tokenizerModel
	}
	if tokenizer == "" {
		return nil, fail("no tokenizer.json or tokenizer.model")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fail("%v", err)
	}
	var hc hfConfig
	if err := json.Unmarshal(raw, &hc); err != nil {
		return nil, fail("config.json: %v", err)
	}

	// Tensors: one canonical blob per large tensor, one packed blob for the
	// small ones.
	base := &baseModel{tensors: make(map[string]safetensors.Tensor, len(all)), tokenizer: tokenizer}
	infos := make([]spec.TensorInfo, 0, len(all))
	var small []safetensors.Entry
	// Report about every tenth of the base, and never more often than every
	// 256 MB, so small models stay quiet.
	reportEvery := max(total/10, 256<<20)
	var done, nextReport int64 = 0, reportEvery
	for _, lt := range all {
		t := lt.t
		base.tensors[t.Name] = t
		infos = append(infos, spec.TensorInfo{Name: t.Name, DType: t.DType, Shape: t.Shape})
		entry := safetensors.Entry{Name: t.Name, DType: t.DType, Shape: t.Shape, Data: lt.file.Data(t)}
		if t.Size < PackThreshold {
			small = append(small, entry)
			continue
		}
		d, err := b.putTensors([]safetensors.Entry{entry}, spec.MediaTypeTensor,
			map[string]string{spec.AnnotationLayer: layer.ID, spec.AnnotationTensor: t.Name})
		if err != nil {
			return nil, fail("tensor %s: %v", t.Name, err)
		}
		b.blobs[layer.ID] = append(b.blobs[layer.ID], d)
		if done += t.Size; done >= nextReport && done < total {
			b.progress(api.Event{Step: step, Message: fmt.Sprintf("%s of %s", api.FormatSize(done), api.FormatSize(total))})
			nextReport = done + reportEvery
		}
	}
	if len(small) > 0 {
		d, err := b.putTensors(small, spec.MediaTypeTensor, map[string]string{spec.AnnotationLayer: layer.ID})
		if err != nil {
			return nil, fail("packed tensors: %v", err)
		}
		b.blobs[layer.ID] = append(b.blobs[layer.ID], d)
	}

	base.arch = spec.Arch{
		Family:      hc.ModelType,
		Fingerprint: spec.ArchFingerprint(infos, tokenizer),
		Blocks:      hc.Layers,
		Hidden:      hc.Hidden,
	}
	var refs []spec.BlobRef
	for _, d := range b.blobs[layer.ID] {
		refs = append(refs, spec.BlobRefFromDescriptor(d))
	}
	layer.Digest = spec.LayerDigest(*layer, refs, nil)
	layer.Chain = spec.ChainID("", layer.Digest)
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("%s, %d blobs, fingerprint %s", familyOr(hc.ModelType), len(b.blobs[layer.ID]), short(base.arch.Fingerprint))})
	return base, nil
}

// putTensors stores entries as one canonical safetensors blob.
func (b *builder) putTensors(entries []safetensors.Entry, mediaType string, annotations map[string]string) (ocispec.Descriptor, error) {
	r, _, err := safetensors.NewReader(entries, nil)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	return b.put(r, mediaType, annotations)
}

// weightFiles lists a model directory's safetensors files: the ones named by
// model.safetensors.index.json when there is one, otherwise every
// *.safetensors file.
func weightFiles(dir string) ([]string, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	switch {
	case err == nil:
		var idx struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(raw, &idx); err != nil {
			return nil, fmt.Errorf("model.safetensors.index.json: %w", err)
		}
		set := map[string]bool{}
		for _, f := range idx.WeightMap {
			if f != filepath.Base(f) || filepath.Ext(f) != ".safetensors" {
				return nil, fmt.Errorf("model.safetensors.index.json: %q is not a safetensors file in the model directory", f)
			}
			set[f] = true
		}
		var names []string
		for f := range set {
			names = append(names, f)
		}
		sort.Strings(names)
		if len(names) == 0 {
			return nil, errors.New("model.safetensors.index.json lists no files")
		}
		return names, nil
	case !errors.Is(err, fs.ErrNotExist):
		return nil, err
	}

	names, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	for i, n := range names {
		names[i] = filepath.Base(n)
	}
	if len(names) > 0 {
		sort.Strings(names)
		return names, nil
	}
	for _, ext := range pickleExts {
		if m, _ := filepath.Glob(filepath.Join(dir, "*"+ext)); len(m) > 0 {
			return nil, fmt.Errorf("found %s but no safetensors: pickle weights can run code when loaded, so convert them to safetensors first", filepath.Base(m[0]))
		}
	}
	return nil, errors.New("no *.safetensors files")
}

func plural(n int, noun string) string {
	if n == 1 {
		return "1 " + noun
	}
	return fmt.Sprintf("%d %ss", n, noun)
}

func familyOr(f string) string {
	if f == "" {
		return "unknown family"
	}
	return f
}

// short abbreviates a digest for messages.
func short(d string) string {
	if len(d) > 19 {
		return d[:19]
	}
	return d
}
