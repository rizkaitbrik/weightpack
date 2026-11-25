package trust

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/sigstore/sigstore-go/pkg/root"
	"github.com/sigstore/sigstore-go/pkg/tuf"
	"github.com/sigstore/sigstore-go/pkg/util"
	"github.com/theupdateframework/go-tuf/v2/metadata/fetcher"
)

// Sigstore reaches a Sigstore instance, by default the public-good one
// that cosign uses: its trusted root (the Fulcio CAs, Rekor logs, CT logs
// and timestamp authorities whose signatures count) and its signing config
// (where those services run). Both come from the instance's TUF
// repository, which is signed and cached on disk, and are fetched on first
// use. Only keyless signing and keyless verification need them.
type Sigstore struct {
	// CacheDir caches the TUF repository; "" means ~/.sigstore/root, which
	// cosign shares.
	CacheDir string
	// Material, when set, is used instead of the TUF repository's trusted
	// root, as for a private Sigstore deployment or in tests.
	Material root.TrustedMaterial

	mu      sync.Mutex
	client  *tuf.Client
	live    *root.LiveTrustedRoot
	signing *root.SigningConfig
}

// refresh is how often a long-running process rereads the trusted root.
const refresh = 24 * time.Hour

func (s *Sigstore) tufOptions() *tuf.Options {
	opts := tuf.DefaultOptions()
	if s.CacheDir != "" {
		opts.CachePath = s.CacheDir
	}
	f := fetcher.NewDefaultFetcher()
	f.SetHTTPUserAgent(util.ConstructUserAgent())
	opts.Fetcher = f
	return opts
}

// TrustedMaterial returns the keys and certificates that keyless
// signatures must chain to.
func (s *Sigstore) TrustedMaterial() (root.TrustedMaterial, error) {
	if s.Material != nil {
		return s.Material, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.live == nil {
		live, err := root.NewLiveTrustedRootFromTargetWithPeriod(s.tufOptions(), "trusted_root.json", refresh)
		if err != nil {
			return nil, fmt.Errorf("fetch Sigstore's trusted root (keyless signatures need it): %w", err)
		}
		s.live = live
	}
	return s.live, nil
}

// SigningConfig returns where the instance's Fulcio, Rekor and timestamp
// authorities run.
func (s *Sigstore) SigningConfig() (*root.SigningConfig, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.signing != nil {
		return s.signing, nil
	}
	if s.client == nil {
		c, err := tuf.New(s.tufOptions())
		if err != nil {
			return nil, fmt.Errorf("reach Sigstore's TUF repository: %w", err)
		}
		s.client = c
	}
	sc, err := root.GetSigningConfig(s.client)
	if err != nil {
		return nil, fmt.Errorf("fetch Sigstore's signing config: %w", err)
	}
	if len(sc.FulcioCertificateAuthorityURLs()) == 0 {
		return nil, errors.New("sigstore's signing config names no Fulcio instance")
	}
	s.signing = sc
	return sc, nil
}

// OIDCIssuer is the identity provider keyless signing logs in with.
func (s *Sigstore) OIDCIssuer() string {
	const publicGood = "https://oauth2.sigstore.dev/auth"
	sc, err := s.SigningConfig()
	if err != nil {
		return publicGood
	}
	svc, err := root.SelectService(sc.OIDCProviderURLs(), []uint32{1}, time.Now())
	if err != nil || svc.URL == "" {
		return publicGood
	}
	return svc.URL
}
