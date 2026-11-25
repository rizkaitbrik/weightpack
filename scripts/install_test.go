// Package scripts tests the install script against a fake release server
// laid out as GitHub's: /latest redirects to /tag/v<version>, and assets
// are under /download/v<version>/.
package scripts

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"testing"
)

const version = "0.9.1"

func target(t *testing.T) string {
	arch := goruntime.GOARCH
	if goruntime.GOOS == "darwin" && arch != "arm64" {
		t.Skip("the installer refuses Intel Macs")
	}
	return fmt.Sprintf("weightpack_%s_%s_%s.tar.gz", version, goruntime.GOOS, arch)
}

// archive is a release tarball with stub wpk and wpkd scripts.
func archive(t *testing.T) []byte {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for _, name := range []string{"wpk", "wpkd", "LICENSE"} {
		body := []byte("#!/bin/sh\necho " + name + " " + version + " \"$@\"\n")
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(body))}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write(body); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func sum(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

const wheel = "weightpack_worker-" + version + "-py3-none-any.whl"

var wheelBytes = []byte("not really a wheel")

// releases serves one release; checksums can be tampered with.
func releases(t *testing.T, tarball []byte, checksums string) *httptest.Server {
	name := target(t)
	mux := http.NewServeMux()
	mux.HandleFunc("/latest", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/tag/v"+version, http.StatusFound)
	})
	mux.HandleFunc("/tag/", func(w http.ResponseWriter, _ *http.Request) { fmt.Fprint(w, "release page") })
	mux.HandleFunc("/download/v"+version+"/"+name, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(tarball) })
	mux.HandleFunc("/download/v"+version+"/"+wheel, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(wheelBytes) })
	mux.HandleFunc("/download/v"+version+"/checksums.txt", func(w http.ResponseWriter, _ *http.Request) { fmt.Fprint(w, checksums) })
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func install(t *testing.T, url string, env ...string) (string, string, error) {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "bin")
	cmd := exec.Command("sh", "install.sh")
	cmd.Env = append(os.Environ(), "WPK_RELEASES="+url, "WPK_INSTALL_DIR="+dir, "WPK_NO_WORKER=1")
	cmd.Env = append(cmd.Env, env...)
	out, err := cmd.CombinedOutput()
	return dir, string(out), err
}

func TestInstallLatest(t *testing.T) {
	tarball := archive(t)
	srv := releases(t, tarball, sum(tarball)+"  "+target(t)+"\n0000  other.tar.gz\n")
	dir, out, err := install(t, srv.URL)
	if err != nil {
		t.Fatalf("install failed: %v\n%s", err, out)
	}
	for _, bin := range []string{"wpk", "wpkd"} {
		info, err := os.Stat(filepath.Join(dir, bin))
		if err != nil || info.Mode()&0o111 == 0 {
			t.Errorf("%s not installed as an executable: %v", bin, err)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "LICENSE")); !os.IsNotExist(err) {
		t.Error("the installer copied more than the binaries")
	}
	for _, want := range []string{"Downloading Weightpack " + version, "Installed wpk and wpkd in " + dir, "not on your PATH", "wpk service install", "wpk doctor"} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
	// WPK_SERVICE runs the installed wpk.
	_, out, err = install(t, srv.URL, "WPK_SERVICE=1", "WPK_VERSION=v"+version)
	if err != nil || !strings.Contains(out, "wpk "+version+" service install") {
		t.Errorf("WPK_SERVICE: %v\n%s", err, out)
	}
}

func TestInstallRefusesABadChecksum(t *testing.T) {
	tarball := archive(t)
	srv := releases(t, tarball, strings.Repeat("ab", 32)+"  "+target(t)+"\n")
	dir, out, err := install(t, srv.URL)
	if err == nil || !strings.Contains(out, "checksum mismatch") {
		t.Fatalf("a tampered archive installed: %v\n%s", err, out)
	}
	if _, err := os.Stat(filepath.Join(dir, "wpk")); !os.IsNotExist(err) {
		t.Error("wpk was installed despite the mismatch")
	}
	srv = releases(t, tarball, "")
	if _, out, err := install(t, srv.URL); err == nil || !strings.Contains(out, "does not list") {
		t.Errorf("missing checksum: %v\n%s", err, out)
	}
}

func TestInstallMissingRelease(t *testing.T) {
	srv := releases(t, archive(t), "")
	if _, out, err := install(t, srv.URL, "WPK_VERSION=0.0.0"); err == nil || !strings.Contains(out, "cannot download") {
		t.Errorf("missing release: %v\n%s", err, out)
	}
}

// With uv on PATH, the installer verifies the worker wheel and installs it
// as a uv tool, with the serve extra.
func TestInstallWorker(t *testing.T) {
	tarball := archive(t)
	sums := sum(tarball) + "  " + target(t) + "\n" + sum(wheelBytes) + "  " + wheel + "\n"
	srv := releases(t, tarball, sums)
	fake := t.TempDir()
	calls := filepath.Join(fake, "calls")
	// The requirement is one argument; the wheel it names must exist.
	stub := "#!/bin/sh\necho \"$@\" >> " + calls + "\ntest -f \"${4#*@ file://}\" || exit 9\n"
	if err := os.WriteFile(filepath.Join(fake, "uv"), []byte(stub), 0o755); err != nil {
		t.Fatal(err)
	}
	_, out, err := install(t, srv.URL, "WPK_NO_WORKER=", "PATH="+fake+":"+os.Getenv("PATH"))
	if err != nil {
		t.Fatalf("install: %v\n%s", err, out)
	}
	raw, _ := os.ReadFile(calls)
	got := strings.TrimSpace(string(raw))
	if !strings.HasPrefix(got, "tool install --force weightpack-worker[serve] @ file://") || !strings.HasSuffix(got, "/"+wheel) {
		t.Errorf("uv called with %q", got)
	}

	// A wheel that does not match its checksum is refused before uv runs.
	srv = releases(t, tarball, sum(tarball)+"  "+target(t)+"\n"+strings.Repeat("cd", 32)+"  "+wheel+"\n")
	_ = os.Remove(calls)
	if _, out, err := install(t, srv.URL, "WPK_NO_WORKER=", "PATH="+fake+":"+os.Getenv("PATH")); err == nil || !strings.Contains(out, "checksum mismatch for "+wheel) {
		t.Errorf("tampered wheel: %v\n%s", err, out)
	}
	if _, err := os.Stat(calls); !os.IsNotExist(err) {
		t.Error("uv ran on a tampered wheel")
	}
}
