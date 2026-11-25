package main

import (
	"bytes"
	"io"
	"log"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ggcr "github.com/google/go-containerregistry/pkg/registry"

	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
)

func TestSignVerifyTrust(t *testing.T) {
	t.Setenv("DOCKER_CONFIG", t.TempDir())
	t.Setenv(passwordEnv, "correct horse")
	for _, v := range []string{"SIGSTORE_ID_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN"} {
		t.Setenv(v, "")
	}
	reg := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(reg.Close)
	host := strings.TrimPrefix(reg.URL, "http://")
	signed, unsigned := host+"/team/assistant:v1", host+"/team/draft:v1"

	keys := t.TempDir()
	alice, mallory := filepath.Join(keys, "alice"), filepath.Join(keys, "mallory")
	var out, errOut bytes.Buffer
	mustRun := func(args ...string) string {
		t.Helper()
		out.Reset()
		errOut.Reset()
		if code := run(args, &out, &errOut); code != 0 {
			t.Fatalf("%v: exit %d\n%s%s", args, code, out.String(), errOut.String())
		}
		return out.String()
	}
	mustFail := func(code int, want string, args ...string) {
		t.Helper()
		out.Reset()
		errOut.Reset()
		if got := run(args, &out, &errOut); got != code || !strings.Contains(errOut.String(), want) {
			t.Errorf("%v: exit %d, want %d with %q\n%s%s", args, got, code, want, out.String(), errOut.String())
		}
	}

	if got := mustRun("keygen", "-o", alice); !strings.Contains(got, "key SHA256:") {
		t.Errorf("keygen output: %s", got)
	}
	mustRun("keygen", "-o", mallory)
	mustFail(1, "exists", "keygen", "-o", alice)
	if info, err := os.Stat(alice + ".key"); err != nil || info.Mode().Perm() != 0o600 {
		t.Errorf("private key mode: %v, %v", info.Mode(), err)
	}

	// Build, push one image signed and one not.
	startDaemon(t)
	dir := t.TempDir()
	buildtest.Project(t, dir)
	mustRun("build", "-t", signed, "-t", unsigned, dir)
	got := mustRun("push", "--sign", "--key", alice+".key", signed)
	if !strings.Contains(got, "Pushed "+signed) || !strings.Contains(got, "Signed "+host+"/team/assistant@sha256:") {
		t.Errorf("push --sign output:\n%s", got)
	}
	mustRun("push", unsigned)
	mustFail(2, "go with --sign", "push", "--key", alice+".key", unsigned)
	// Keyless signing without a token, in a test with no terminal, says
	// how to get one, before pushing anything.
	mustFail(1, "needs an OIDC token", "push", "--sign", unsigned)

	if got := mustRun("verify", "--key", alice+".pub", signed); !strings.Contains(got, "Verified "+signed+"@sha256:") || !strings.Contains(got, "signed by key SHA256:") {
		t.Errorf("verify output: %s", got)
	}
	mustFail(1, "not one of the trusted keys", "verify", "--key", mallory+".pub", signed)
	mustFail(1, "no Sigstore signature", "verify", "--key", alice+".pub", unsigned)
	mustFail(2, "exactly one of --issuer", "verify", "--identity", "a@example.com", signed)

	// wpk sign signs an image already in the registry.
	t.Setenv(passwordEnv, "wrong")
	mustFail(1, "wrong password", "sign", "--key", mallory+".key", unsigned)
	t.Setenv(passwordEnv, "correct horse")
	if got := mustRun("sign", "--key", mallory+".key", unsigned); !strings.Contains(got, "Signed "+host+"/team/draft@") {
		t.Errorf("sign output: %s", got)
	}

	// A second machine pulls under a trust policy.
	startDaemon(t)
	if got := mustRun("trust"); !strings.Contains(got, "No trust rules") {
		t.Errorf("empty policy: %s", got)
	}
	mustRun("trust", "set", host, "--key", alice+".pub")
	mustRun("trust", "set", "*", "--reject")
	got = mustRun("trust", "ls")
	if !strings.Contains(got, host+"  signed  key SHA256:") || !strings.Contains(got, "*") {
		t.Errorf("trust ls:\n%s", got)
	}
	if got := mustRun("verify", signed); !strings.Contains(got, "trust rule "+host) {
		t.Errorf("verify against the policy: %s", got)
	}
	mustFail(1, "none from a trusted signer", "pull", unsigned)
	got = mustRun("pull", signed)
	if !strings.Contains(got, "Verified sha256:") || !strings.Contains(got, "Tagged "+signed) {
		t.Errorf("pull of a signed image:\n%s", got)
	}
	// A narrower rule wins.
	mustRun("trust", "set", host+"/team/draft", "--key", mallory+".pub")
	mustRun("pull", unsigned)
	mustRun("trust", "rm", host)
	mustFail(1, "rejects images from *", "pull", signed)
	mustFail(1, "no rule for", "trust", "rm", host)
	mustFail(2, "give signers", "trust", "set", host, "--accept", "--reject")
}
