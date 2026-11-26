package daemon

import (
	"context"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/hf"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// lowDisk is the free space below which doctor warns: a single 8B base in
// BF16 is 16 GB, and serving exports one more copy.
const lowDisk = 40_000_000_000

// doctor checks what wpkd depends on, in its own environment: that is the
// one builds and engines run in.
func (s *server) doctor(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	checks := []api.Check{{Name: "wpkd", Status: api.CheckOK, Detail: fmt.Sprintf("version %s, store %s", s.version, s.store.Root())}}
	checks = append(checks, s.checkWorker(ctx))
	if s.runtime != nil {
		checks = append(checks, s.runtime.Doctor(ctx)...)
		checks = append(checks, api.Check{Name: "gateway", Status: api.CheckOK, Detail: s.runtime.Endpoint()})
	} else {
		checks = append(checks, api.Check{Name: "gateway", Status: api.CheckWarn, Detail: "serving is off", Fix: "start wpkd without --gateway '' to serve images"})
	}
	checks = append(checks, s.checkStore(), s.checkHF())
	writeJSON(w, http.StatusOK, api.DoctorReport{Checks: checks})
}

func (s *server) checkWorker(ctx context.Context) api.Check {
	c := api.Check{Name: "build worker"}
	if s.worker == nil {
		c.Status, c.Detail = api.CheckFail, "none configured"
		c.Fix = `restart wpkd with --worker weightpack-worker (pip install weightpack-worker)`
		return c
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	info, err := s.worker.Info(ctx, &workerpb.InfoRequest{})
	if err != nil {
		c.Status, c.Detail = api.CheckFail, fmt.Sprintf("no answer on %s: %v", s.worker.Socket(), err)
		c.Fix = "restart wpkd with --worker weightpack-worker, or start the worker on that socket"
		return c
	}
	var codecs []string
	for _, cd := range info.Codecs {
		codecs = append(codecs, strings.ToLower(strings.TrimPrefix(cd.String(), "CODEC_")))
	}
	c.Status = api.CheckOK
	c.Detail = fmt.Sprintf("weightpack-worker %s on %s (codecs: %s)", info.Version, s.worker.Socket(), strings.Join(codecs, ", "))
	return c
}

func (s *server) checkStore() api.Check {
	used := dirSize(filepath.Join(s.store.Root(), "blobs"))
	c := api.Check{Name: "store", Status: api.CheckOK}
	var st syscall.Statfs_t
	if err := syscall.Statfs(s.store.Root(), &st); err != nil {
		c.Detail = fmt.Sprintf("%s of blobs; free space unknown: %v", api.FormatSize(used), err)
		return c
	}
	free := int64(st.Bavail) * int64(st.Bsize)
	c.Detail = fmt.Sprintf("%s of blobs; %s free", api.FormatSize(used), api.FormatSize(free))
	if free < lowDisk {
		c.Status = api.CheckWarn
		c.Fix = "free some space: wpk rm unused images, then wpk prune (--downloads also drops the hf:// cache)"
	}
	return c
}

func (s *server) checkHF() api.Check {
	c := api.Check{Name: "hugging face"}
	switch {
	case s.hf == nil:
		c.Status, c.Detail = api.CheckSkip, "hf:// sources are off"
	case s.hf.Token == "":
		c.Status, c.Detail = api.CheckWarn, "no token: public repositories only"
		c.Fix = "for gated or private models (Llama, Gemma), run hf auth login or set HF_TOKEN, then restart wpkd"
	default:
		endpoint := s.hf.Endpoint
		if endpoint == "" {
			endpoint = hf.DefaultEndpoint
		}
		c.Status, c.Detail = api.CheckOK, "token set; endpoint "+endpoint
	}
	return c
}
