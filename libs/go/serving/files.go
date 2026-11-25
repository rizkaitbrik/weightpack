package serving

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build"
	"github.com/rizkaitbrik/weightpack/libs/go/export"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/squash"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// WriteServingFiles writes under dir, which must not exist, the files an
// engine serves an image from, as Run prepares them: the serving base, baked
// through the layers below the adapters when there are any, and each
// adapter. ROUTER train reads them, so a router learns on what is served.
// The caller removes dir.
func WriteServingFiles(ctx context.Context, s *store.Store, w workerpb.WorkerClient, image, dir string, progress func(api.Event)) (*build.ServingFiles, error) {
	root, err := s.Resolve(image)
	if err != nil {
		return nil, err
	}
	var m ocispec.Manifest
	var cfg spec.Config
	if err := s.ReadJSON(root.Digest.String(), &m); err != nil {
		return nil, err
	}
	if err := s.ReadJSON(m.Config.Digest.String(), &cfg); err != nil {
		return nil, err
	}
	p, err := planFor(&cfg)
	if err != nil {
		return nil, err
	}
	src := root.Digest.String()
	if p.through != "" {
		if w == nil {
			return nil, errors.New("baking the serving base needs wpkd's build worker")
		}
		progress(api.Event{Step: "serving files", Message: fmt.Sprintf("baking the layers through %s into the serving base", p.through)})
		res, err := squash.Squash(ctx, s, w, src, squash.Options{Through: p.through}, progress)
		if err != nil {
			return nil, err
		}
		src = res.Manifest.Digest.String()
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	base := filepath.Join(dir, "base")
	if _, err := export.Export(ctx, s, src, export.Options{Dir: base, Adapters: []string{}}, progress); err != nil {
		return nil, err
	}
	out := &build.ServingFiles{Base: filepath.Join(base, "base"), Adapters: map[string]string{}}
	if len(p.adapters) == 0 {
		return out, nil
	}
	ids := make([]string, len(p.adapters))
	for i, a := range p.adapters {
		ids[i] = a.ID
	}
	adapters := filepath.Join(dir, "adapters")
	if _, err := export.Export(ctx, s, root.Digest.String(), export.Options{Dir: adapters, NoBase: true, Adapters: ids}, progress); err != nil {
		return nil, err
	}
	for _, id := range ids {
		out.Adapters[id] = filepath.Join(adapters, "adapters", id)
	}
	return out, nil
}
