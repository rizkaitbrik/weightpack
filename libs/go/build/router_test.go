package build

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"google.golang.org/grpc"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// routed are the modules buildtest's MIX domain (targets=mlp blocks=1)
// routes in, with their input sizes.
var routed = map[string]int64{
	"model.layers.1.mlp.gate_proj": buildtest.Hidden,
	"model.layers.1.mlp.up_proj":   buildtest.Hidden,
	"model.layers.1.mlp.down_proj": buildtest.Inter,
}

// writeRouter writes a router file for modules with n experts.
func writeRouter(t *testing.T, path string, modules map[string]int64, n int64) {
	t.Helper()
	var ts []buildtest.Tensor
	seed := int64(0)
	for m, in := range modules {
		seed++
		ts = append(ts,
			buildtest.Tensor{Name: m + routerWeight, Shape: []int64{n, in}, Seed: seed},
			buildtest.Tensor{Name: m + routerBias, Shape: []int64{n}, Seed: seed + 100},
		)
	}
	buildtest.WriteSafetensors(t, path, ts)
}

func withRouter(line string) string {
	return strings.Replace(buildtest.Assistant, "ROUTER arrow", line, 1)
}

func mixOf(c *spec.Config) spec.Layer {
	for _, l := range c.Layers {
		if l.Kind == spec.KindMix {
			return l
		}
	}
	return spec.Layer{}
}

func TestRouterFile(t *testing.T) {
	f := newFixture(t)
	writeRouter(t, filepath.Join(f.dir, "router.safetensors"), routed, 2)
	arrow, err := f.build(t, f.weightfile(t, "Weightfile.arrow", buildtest.Assistant), "arrow")
	if err != nil {
		t.Fatal(err)
	}
	wf := f.weightfile(t, "Weightfile.trained", withRouter("ROUTER ./router.safetensors"))
	res, err := f.build(t, wf, "trained")
	if err != nil {
		t.Fatal(err)
	}
	m, c := f.load(t, "trained")
	mix := mixOf(c)
	if mix.Router != spec.RouterTrained || mix.Chain == mixOf(arrow.Config).Chain {
		t.Errorf("mix = %+v", mix)
	}
	var blobs int
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationLayer] == "domain" {
			blobs++
			if d.MediaType != spec.MediaTypeRouter || d.Annotations[spec.AnnotationTitle] != spec.RouterFile {
				t.Errorf("router blob %+v", d)
			}
		}
	}
	if blobs != 1 {
		t.Errorf("%d router blobs", blobs)
	}
	again, err := f.build(t, wf)
	if err != nil || again.Manifest.Digest != res.Manifest.Digest {
		t.Errorf("rebuild: %v %v, want %v", err, again.Manifest.Digest, res.Manifest.Digest)
	}

	extra := map[string]int64{"model.layers.0.mlp.up_proj": buildtest.Hidden}
	for k, v := range routed {
		extra[k] = v
	}
	missing := map[string]int64{"model.layers.1.mlp.up_proj": buildtest.Hidden}
	for name, c := range map[string]struct {
		modules map[string]int64
		n       int64
		want    string
	}{
		"extra":   {extra, 2, "model.layers.0.mlp.up_proj is not a module MIX domain routes in"},
		"missing": {missing, 2, "no router weight or bias for 2 modules"},
		"rows":    {routed, 3, "want F32 [2"},
	} {
		p := filepath.Join(f.dir, name+".safetensors")
		writeRouter(t, p, c.modules, c.n)
		_, err := f.build(t, f.weightfile(t, "Weightfile."+name, withRouter("ROUTER ./"+name+".safetensors")))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", name, err, c.want)
		}
	}
}

// trainWorker stands in for the Python worker's TrainRouter: it writes a
// router for the modules it is told, as training would.
type trainWorker struct {
	workerpb.WorkerClient
	t       *testing.T
	calls   int
	req     *workerpb.TrainRouterRequest
	version string
}

func (w *trainWorker) Info(context.Context, *workerpb.InfoRequest, ...grpc.CallOption) (*workerpb.InfoResponse, error) {
	return &workerpb.InfoResponse{Version: w.version}, nil
}

func (w *trainWorker) TrainRouter(_ context.Context, req *workerpb.TrainRouterRequest, _ ...grpc.CallOption) (grpc.ServerStreamingClient[workerpb.TrainRouterProgress], error) {
	w.calls++
	w.req = req
	writeRouter(w.t, req.OutFile, routed, int64(len(req.Experts)))
	events := []*workerpb.TrainRouterProgress{
		{Step: 5, Steps: 10, Loss: 2},
		{Step: 10, Steps: 10, Loss: 1},
		{Done: true, Examples: 9, Heldout: 1, ArrowLoss: 2.5, TrainedLoss: 1.5, Device: "cpu", Seconds: 3},
	}
	return &progressStream{events: events}, nil
}

type progressStream struct {
	grpc.ClientStream
	events []*workerpb.TrainRouterProgress
}

func (s *progressStream) Recv() (*workerpb.TrainRouterProgress, error) {
	if len(s.events) == 0 {
		return nil, io.EOF
	}
	e := s.events[0]
	s.events = s.events[1:]
	return e, nil
}

func TestRouterTrain(t *testing.T) {
	f := newFixture(t)
	if err := os.WriteFile(filepath.Join(f.dir, "mix.jsonl"), []byte(`{"text": "a"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	w := &trainWorker{t: t, version: "1"}
	var trainedOn string
	serve := func(_ context.Context, image, dir string, _ func(api.Event)) (*ServingFiles, error) {
		trainedOn = image
		return &ServingFiles{Base: filepath.Join(dir, "base"), Adapters: map[string]string{
			"legal": filepath.Join(dir, "legal"), "medical": filepath.Join(dir, "medical"),
		}}, nil
	}
	wf := f.weightfile(t, "Weightfile.train", withRouter("ROUTER train data=./mix.jsonl steps=10"))
	build := func() (*api.BuildResult, error) {
		f.events = nil
		return Build(context.Background(), f.store, Options{Weightfile: wf, Worker: w, ServingFiles: serve, Tags: []string{"trained"}},
			func(e api.Event) { f.events = append(f.events, e) })
	}
	res, err := build()
	if err != nil {
		t.Fatal(err)
	}
	r := w.req
	if w.calls != 1 || r.TopK != 1 || r.Blocks != "1" || r.Steps != 10 || !slices.Equal(r.Targets, []string{"gate", "up", "down"}) ||
		r.Data != filepath.Join(f.dir, "mix.jsonl") || r.ChatTemplate != filepath.Join(f.dir, "chat.jinja") ||
		len(r.Experts) != 2 || r.Experts[0].Id != "legal" || r.Experts[1].Dir != filepath.Join(filepath.Dir(r.BaseDir), "medical") {
		t.Errorf("request = %+v", r)
	}
	// Training ran on the same image with Arrow's router.
	_, arrow := f.load(t, trainedOn)
	if mixOf(arrow).Router != spec.RouterArrow {
		t.Errorf("trained on %+v", mixOf(arrow))
	}
	if mixOf(res.Config).Router != spec.RouterTrained {
		t.Errorf("mix = %+v", mixOf(res.Config))
	}
	if !slices.ContainsFunc(f.events, func(e api.Event) bool {
		return strings.Contains(e.Message, "held-out loss 1.5000 with the trained router, 2.5000 with Arrow")
	}) {
		t.Errorf("events = %+v", f.events)
	}

	// The same chain, data and worker: the router comes from the cache.
	again, err := build()
	if err != nil || w.calls != 1 || again.Manifest.Digest != res.Manifest.Digest {
		t.Errorf("rebuild: %v, %d calls, %v want %v", err, w.calls, again.Manifest.Digest, res.Manifest.Digest)
	}
	// A new worker version trains again.
	w.version = "2"
	if _, err := build(); err != nil || w.calls != 2 {
		t.Errorf("new worker: %v, %d calls", err, w.calls)
	}

	_, err = Build(context.Background(), f.store, Options{Weightfile: wf}, nil)
	if err == nil || !strings.Contains(err.Error(), "needs wpkd's build worker") {
		t.Errorf("no worker: %v", err)
	}
}
