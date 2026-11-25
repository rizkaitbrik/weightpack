package build

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"google.golang.org/grpc"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// lossyWorker stands in for the worker's lossy EncodeLayer and Evaluate: it
// writes each changed tensor's delta file in the codec's layout (the build
// checks layouts, not values) and answers evals with fixed scores.
type lossyWorker struct {
	workerpb.WorkerClient
	encodes, evals int
	req            *workerpb.EncodeLayerRequest
	evalReq        *workerpb.EvaluateRequest
	source, score  float64
	corrupt        bool // write a bit1 signs tensor one byte too long
}

func (w *lossyWorker) Info(context.Context, *workerpb.InfoRequest, ...grpc.CallOption) (*workerpb.InfoResponse, error) {
	return &workerpb.InfoResponse{Version: "1"}, nil
}

func (w *lossyWorker) EncodeLayer(_ context.Context, req *workerpb.EncodeLayerRequest, _ ...grpc.CallOption) (*workerpb.EncodeLayerResponse, error) {
	w.encodes++
	w.req = req
	var basePaths []string
	for _, b := range req.Base {
		basePaths = append(basePaths, b.Path)
	}
	parent, info, err := tensorBytes(basePaths)
	if err != nil {
		return nil, err
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
	resp := &workerpb.EncodeLayerResponse{RelativeError: 0.3}
	for i, n := range names {
		if bytes.Equal(target[n], parent[n]) {
			resp.Unchanged++
			continue
		}
		shape := info[n].Shape
		count := int64(1)
		for _, d := range shape {
			count *= d
		}
		type e struct {
			suffix, dtype string
			shape         []int64
		}
		var es []e
		switch {
		case len(shape) < 2:
			es = []e{{lossyDelta, "F32", shape}}
		case req.Codec == workerpb.Codec_CODEC_BIT1:
			signs := (count + 7) / 8
			if w.corrupt {
				signs++
			}
			es = []e{{lossySigns, "U8", []int64{signs}}, {lossyScale, "F32", []int64{1}}}
		default:
			k := int64(math.Ceil(req.Density * float64(count)))
			es = []e{{lossyIndices, "U32", []int64{k}}, {lossyValues, "F32", []int64{k}}}
		}
		var entries []safetensors.Entry
		for _, x := range es {
			size, _ := safetensors.ByteSize(x.dtype, x.shape)
			entries = append(entries, safetensors.Entry{Name: n + x.suffix, DType: x.dtype, Shape: x.shape, Data: bytes.NewReader(make([]byte, size))})
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
		r, _, err := safetensors.NewReader(entries, nil)
		if err != nil {
			return nil, err
		}
		out, _ := io.ReadAll(r)
		file := fmt.Sprintf("%d.safetensors", i)
		if err := os.WriteFile(filepath.Join(req.OutDir, file), out, 0o600); err != nil {
			return nil, err
		}
		resp.Tensors = append(resp.Tensors, &workerpb.EncodedTensor{Name: n, File: file})
	}
	return resp, nil
}

func (w *lossyWorker) Evaluate(_ context.Context, req *workerpb.EvaluateRequest, _ ...grpc.CallOption) (*workerpb.EvaluateResponse, error) {
	w.evals++
	w.evalReq = req
	name := "self"
	if req.Suite != "" {
		name = strings.TrimSuffix(filepath.Base(req.Suite), filepath.Ext(req.Suite))
	}
	return &workerpb.EvaluateResponse{Suite: name, Examples: 24, Tokens: 3000, Scores: []*workerpb.ModelScore{
		{Accuracy: w.source, Loss: 1}, {Accuracy: w.score, Loss: 1.1},
	}}, nil
}

func (f *fixture) buildLossy(t *testing.T, name, body string, w *lossyWorker) (*api.BuildResult, error) {
	t.Helper()
	f.events = nil
	return Build(context.Background(), f.store, Options{Weightfile: f.weightfile(t, name, body), Worker: w}, func(e api.Event) {
		f.events = append(f.events, e)
	})
}

func TestLossyLayer(t *testing.T) {
	f := newFixture(t)
	writeFinetune(t, filepath.Join(f.dir, "finance"), 7)
	w := &lossyWorker{source: 90, score: 89.5}
	body := "FROM ./base\nLAYER finance ./finance codec=bit1 max_drop=1\n"
	res, err := f.buildLossy(t, "Weightfile", body, w)
	if err != nil {
		t.Fatal(err)
	}
	m, c := f.load(t, res.Manifest.Digest.String())
	if err := spec.CheckImage(m, c); err != nil {
		t.Fatal(err)
	}
	fin := c.Layers[1]
	fid := fin.Fidelity
	if fin.Kind != spec.KindDeltaBit1 || fid == nil || fid.Codec != "bit1" || fid.Source != "./finance" ||
		!strings.HasPrefix(fid.SourceDigest, "sha256:") || fid.Eval != "self" || fid.EvalDigest != "" ||
		*fid.SourceScore != 90 || *fid.Score != 89.5 || *fid.MeasuredDrop != 0.5 || fid.MaxDrop != 1 {
		t.Fatalf("layer = %+v, fidelity %+v", fin, fid)
	}
	blobs := 0
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationLayer] == "finance" {
			blobs++
			if d.MediaType != spec.MediaTypeDeltaBit1 || d.Annotations[spec.AnnotationTensor] == "" {
				t.Errorf("blob %+v", d)
			}
		}
	}
	if blobs != 4 { // the fine-tune changes q and v in both blocks
		t.Errorf("%d delta blobs", blobs)
	}
	if w.req.Codec != workerpb.Codec_CODEC_BIT1 || w.req.ReconstructDir == "" || w.evalReq.Suite != "" ||
		!slices.Equal(w.evalReq.Models, []string{filepath.Join(f.dir, "finance"), w.req.ReconstructDir}) {
		t.Errorf("encode %+v, eval %+v", w.req, w.evalReq)
	}
	if !slices.ContainsFunc(f.events, func(e api.Event) bool { return strings.Contains(e.Message, "a drop of 0.50 points (max_drop 1)") }) {
		t.Errorf("events = %+v", f.events)
	}

	// A rebuild takes the deltas and the eval from the cache.
	again, err := f.buildLossy(t, "Weightfile", body, w)
	if err != nil || w.encodes != 1 || w.evals != 1 || again.Manifest.Digest != res.Manifest.Digest {
		t.Errorf("rebuild: %v, %d encodes, %d evals", err, w.encodes, w.evals)
	}
	// A tighter gate fails on the cached drop.
	_, err = f.buildLossy(t, "Weightfile", "FROM ./base\nLAYER finance ./finance codec=bit1 max_drop=0.2\n", w)
	if err == nil || !strings.Contains(err.Error(), "drop on self is 0.50 points, more than max_drop=0.2") {
		t.Errorf("tight gate: %v", err)
	}
}

func TestSparseLayerOnASuite(t *testing.T) {
	f := newFixture(t)
	writeFinetune(t, filepath.Join(f.dir, "finance"), 7)
	if err := os.WriteFile(filepath.Join(f.dir, "finance.jsonl"), []byte(`{"text": "a"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	w := &lossyWorker{source: 70, score: 70.2}
	res, err := f.buildLossy(t, "Weightfile", "FROM ./base\nLAYER finance ./finance codec=sparse density=0.25 eval=./finance.jsonl max_drop=0\n", w)
	if err != nil {
		t.Fatal(err)
	}
	fid := res.Config.Layers[1].Fidelity
	if res.Config.Layers[1].Kind != spec.KindDeltaSparse || fid.Density != 0.25 || fid.Eval != "finance" ||
		!strings.HasPrefix(fid.EvalDigest, "sha256:") || *fid.MeasuredDrop >= 0 {
		t.Errorf("fidelity %+v", fid)
	}
	if w.req.Density != 0.25 || w.evalReq.Suite != filepath.Join(f.dir, "finance.jsonl") {
		t.Errorf("encode %+v, eval %+v", w.req, w.evalReq)
	}
}

func TestLossyLayerRefuses(t *testing.T) {
	f := newFixture(t)
	writeFinetune(t, filepath.Join(f.dir, "finance"), 7)
	for name, c := range map[string]struct {
		body string
		w    *lossyWorker
		want string
	}{
		"bad output": {"FROM ./base\nLAYER finance ./finance codec=bit1 max_drop=1\n", &lossyWorker{corrupt: true}, "worker output for model.layers.0.self_attn.q_proj.weight"},
		"over LoRA": {"FROM ./base\nLAYER french ./adapters/french\nLAYER finance ./finance codec=bit1 max_drop=1\n", &lossyWorker{},
			"a codec=bit1 layer sits only on the base or on XOR layers, but french is a lora layer"},
		"too lossy": {"FROM ./base\nLAYER finance ./finance codec=sparse max_drop=2\n", &lossyWorker{source: 80, score: 60}, "drop on self is 20.00 points, more than max_drop=2"},
	} {
		if _, err := f.buildLossy(t, name+".Weightfile", c.body, c.w); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", name, err, c.want)
		}
	}
}
