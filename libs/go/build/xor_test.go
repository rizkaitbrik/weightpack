package build

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"google.golang.org/grpc"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// xorWorker stands in for the Python worker's EncodeLayer: plain byte XOR
// of each target tensor with the parent's (base, then parent deltas).
type xorWorker struct {
	workerpb.WorkerClient
	req *workerpb.EncodeLayerRequest
}

func tensorBytes(paths []string) (map[string][]byte, map[string]safetensors.Tensor, error) {
	data, info := map[string][]byte{}, map[string]safetensors.Tensor{}
	for _, p := range paths {
		f, err := safetensors.Open(p)
		if err != nil {
			return nil, nil, err
		}
		for _, t := range f.Tensors {
			b, err := io.ReadAll(f.Data(t))
			if err != nil {
				return nil, nil, err
			}
			data[t.Name], info[t.Name] = b, t
		}
		_ = f.Close()
	}
	return data, info, nil
}

func xorInto(dst, src []byte) {
	for i := range dst {
		dst[i] ^= src[i]
	}
}

func (w *xorWorker) EncodeLayer(_ context.Context, req *workerpb.EncodeLayerRequest, _ ...grpc.CallOption) (*workerpb.EncodeLayerResponse, error) {
	w.req = req
	var basePaths []string
	for _, b := range req.Base {
		basePaths = append(basePaths, b.Path)
	}
	parent, info, err := tensorBytes(basePaths)
	if err != nil {
		return nil, err
	}
	for _, l := range req.Parents {
		var ps []string
		for _, b := range l.Blobs {
			ps = append(ps, b.Path)
		}
		deltas, _, err := tensorBytes(ps)
		if err != nil {
			return nil, err
		}
		for n, d := range deltas {
			xorInto(parent[n], d)
		}
	}
	target, _, err := tensorBytes(req.TargetFiles)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(target))
	for n := range target {
		names = append(names, n)
	}
	sort.Strings(names)
	resp := &workerpb.EncodeLayerResponse{}
	for i, n := range names {
		if bytes.Equal(target[n], parent[n]) {
			resp.Unchanged++
			continue
		}
		xorInto(parent[n], target[n])
		r, _, _ := safetensors.NewReader([]safetensors.Entry{{Name: n, DType: info[n].DType, Shape: info[n].Shape, Data: bytes.NewReader(parent[n])}}, nil)
		out, _ := io.ReadAll(r)
		file := fmt.Sprintf("%d.safetensors", i)
		if err := os.WriteFile(filepath.Join(req.OutDir, file), out, 0o600); err != nil {
			return nil, err
		}
		resp.Tensors = append(resp.Tensors, &workerpb.EncodedTensor{Name: n, File: file})
	}
	return resp, nil
}

// writeFinetune writes a "fine-tune" of the fixture base: the base with the
// low bytes of the q and v matrices nudged, everything else unchanged.
func writeFinetune(t *testing.T, dir string, seed int64) {
	t.Helper()
	var ts []buildtest.Tensor
	for _, x := range buildtest.BaseTensors() {
		if strings.Contains(x.Name, "q_proj") || strings.Contains(x.Name, "v_proj") {
			x.Seed += seed
		}
		ts = append(ts, x)
	}
	buildtest.WriteBase(t, dir, 1)
	buildtest.WriteSafetensors(t, filepath.Join(dir, "model.safetensors"), ts)
}

func (f *fixture) buildXOR(t *testing.T, name, body string, w *xorWorker) (*api.BuildResult, error) {
	t.Helper()
	p := f.weightfile(t, name, body)
	return Build(context.Background(), f.store, Options{Weightfile: p, Worker: w}, nil)
}

func TestXORLayer(t *testing.T) {
	f := newFixture(t)
	writeFinetune(t, filepath.Join(f.dir, "finance"), 7)
	w := &xorWorker{}
	res, err := f.buildXOR(t, "Weightfile", "FROM ./base\nLAYER finance ./finance codec=xor\n", w)
	if err != nil {
		t.Fatal(err)
	}
	m, c := f.load(t, res.Manifest.Digest.String())
	if err := spec.CheckImage(m, c); err != nil {
		t.Fatal(err)
	}
	fin := c.Layers[1]
	if fin.Kind != spec.KindDeltaXOR || fin.Fidelity != nil {
		t.Errorf("finance = %+v, want a delta.xor layer with no fidelity record", fin)
	}
	// Only the four changed matrices are stored, compressed.
	var xor int
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationLayer] == "finance" {
			xor++
			if d.MediaType != spec.MediaTypeDeltaXOR+"+zstd" || !spec.IsLossless(d.MediaType) {
				t.Errorf("blob media type %s", d.MediaType)
			}
			if !strings.Contains(d.Annotations[spec.AnnotationTensor], "_proj") {
				t.Errorf("unchanged tensor stored: %s", d.Annotations[spec.AnnotationTensor])
			}
		}
	}
	if xor != 4 {
		t.Errorf("%d XOR blobs, want 4", xor)
	}

	// Rebuilding gives the same digest: compression is deterministic.
	again, err := f.buildXOR(t, "Weightfile", "FROM ./base\nLAYER finance ./finance codec=xor\n", w)
	if err != nil || again.Manifest.Digest != res.Manifest.Digest {
		t.Errorf("rebuild: %v, digest %s vs %s", err, again.Manifest.Digest, res.Manifest.Digest)
	}

	// A second XOR layer gets the first one's deltas, decoded, as its parent.
	writeFinetune(t, filepath.Join(f.dir, "finance2"), 9)
	if _, err := f.buildXOR(t, "Two.Weightfile", "FROM ./base\nLAYER finance ./finance codec=xor\nLAYER more ./finance2 codec=xor\n", w); err != nil {
		t.Fatal(err)
	}
	if len(w.req.Parents) != 1 || w.req.Parents[0].Id != "finance" || len(w.req.Parents[0].Blobs) != 4 {
		t.Errorf("parents = %+v", w.req.Parents)
	}
	for _, b := range w.req.Parents[0].Blobs {
		if _, err := safetensors.Open(b.Path); err == nil {
			t.Error("decoded parent files should be gone after the build")
		}
	}

	// An unchanged fine-tune stores nothing.
	same, err := f.buildXOR(t, "Same.Weightfile", "FROM ./base\nLAYER same ./base codec=xor\n", w)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(f.blobsOf(t, same, "same")); n != 0 {
		t.Errorf("an unchanged fine-tune stored %d blobs", n)
	}
}

func (f *fixture) blobsOf(t *testing.T, res *api.BuildResult, layer string) []string {
	t.Helper()
	m, _ := f.load(t, res.Manifest.Digest.String())
	var out []string
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationLayer] == layer {
			out = append(out, d.Digest.String())
		}
	}
	return out
}

func TestXORLayerRefuses(t *testing.T) {
	f := newFixture(t)
	writeFinetune(t, filepath.Join(f.dir, "finance"), 7)
	// A fine-tune of another model: one tensor with another shape.
	other := filepath.Join(f.dir, "other")
	buildtest.WriteBase(t, other, 1)
	buildtest.WriteSafetensors(t, filepath.Join(other, "model.safetensors"), []buildtest.Tensor{{Name: "model.norm.weight", Shape: []int64{7}, Seed: 1}})
	cases := map[string]struct{ body, want string }{
		"over a LoRA":   {"FROM ./base\nLAYER french ./adapters/french\nLAYER finance ./finance codec=xor\n", "french is a lora layer"},
		"another model": {"FROM ./base\nLAYER x ./other codec=xor\n", "which the base does not have"},
	}
	for name, c := range cases {
		if _, err := f.buildXOR(t, "Bad.Weightfile", c.body, &xorWorker{}); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", name, err, c.want)
		}
	}
	// LoRA layers can sit on an XOR layer.
	if _, err := f.buildXOR(t, "Ok.Weightfile", "FROM ./base\nLAYER finance ./finance codec=xor\nLAYER french ./adapters/french\n", &xorWorker{}); err != nil {
		t.Errorf("LoRA over XOR: %v", err)
	}
}

// A layer id comes from an image's config, which a pulled image controls:
// decoding must never let it pick where files go.
func TestDecodeXORLayerStaysInItsDirectory(t *testing.T) {
	f := newFixture(t)
	writeFinetune(t, filepath.Join(f.dir, "finance"), 7)
	res, err := f.buildXOR(t, "Weightfile", "FROM ./base\nLAYER finance ./finance codec=xor\n", &xorWorker{})
	if err != nil {
		t.Fatal(err)
	}
	m, _ := f.load(t, res.Manifest.Digest.String())
	var blobs []ocispec.Descriptor
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationLayer] == "finance" {
			blobs = append(blobs, d)
		}
	}
	root := t.TempDir()
	dir := filepath.Join(root, "a", "b")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	x, err := DecodeXORLayer(f.store, "../../escaped", blobs, dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(x.Blobs) != len(blobs) {
		t.Fatalf("decoded %d of %d blobs", len(x.Blobs), len(blobs))
	}
	for _, b := range x.Blobs {
		if filepath.Dir(b.Path) != dir {
			t.Errorf("decoded file %s is outside %s", b.Path, dir)
		}
	}
	if leaked, _ := filepath.Glob(filepath.Join(root, "escaped*")); len(leaked) > 0 {
		t.Errorf("files escaped the directory: %v", leaked)
	}
}

func TestXORChecksTheTokenizer(t *testing.T) {
	f := newFixture(t)
	cases := map[string]struct {
		tokenizer string // "" removes it
		want      string // "" means the import succeeds
	}{
		"same bytes":       {`{"model":{"type":"BPE"}}`, ""},
		"reformatted":      {"{ \"version\": \"1.0\",\n  \"model\" : { \"type\" : \"BPE\" } }", ""},
		"other vocabulary": {`{"model":{"type":"BPE","vocab":{"hello":0}}}`, "tokenizer differs from its parent's"},
		"none":             {"", ""},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(f.dir, "ft-"+strings.ReplaceAll(name, " ", "-"))
			writeFinetune(t, dir, 7)
			tok := filepath.Join(dir, "tokenizer.json")
			if c.tokenizer == "" {
				_ = os.Remove(tok)
			} else if err := os.WriteFile(tok, []byte(c.tokenizer), 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := f.buildXOR(t, "W", "FROM ./base\nLAYER ft "+dir+" codec=xor\n", &xorWorker{})
			switch {
			case c.want == "" && err != nil:
				t.Fatalf("import refused: %v", err)
			case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
				t.Fatalf("err = %v, want %q", err, c.want)
			}
		})
	}
}
