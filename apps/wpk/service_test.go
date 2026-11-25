package main

import (
	"bytes"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/service"
)

func fakeServices(t *testing.T) (home string, calls *[]string) {
	t.Helper()
	home = t.TempDir()
	var log []string
	old := serviceManager
	serviceManager = func() (service.Manager, error) {
		return service.Manager{OS: goruntime.GOOS, Home: home, UID: 501, Run: func(name string, args ...string) ([]byte, error) {
			log = append(log, name+" "+strings.Join(args, " "))
			return nil, nil
		}}, nil
	}
	t.Cleanup(func() { serviceManager = old })
	return home, &log
}

func TestServiceDryRun(t *testing.T) {
	if goruntime.GOOS != "darwin" && goruntime.GOOS != "linux" {
		t.Skip("no service manager here")
	}
	_, calls := fakeServices(t)
	t.Setenv("PATH", "/opt/bin:/usr/bin")
	var out, errOut bytes.Buffer
	root := t.TempDir()
	code := run([]string{"service", "install", "--dry-run", "--wpkd", "/opt/bin/wpkd", "--worker", "uv run weightpack-worker", "--root", root, "--gateway", "0.0.0.0:8090", "--gateway-key", "k"}, &out, &errOut)
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	want := []string{"/opt/bin/wpkd", "uv run weightpack-worker", root, "0.0.0.0:8090", "/opt/bin:/usr/bin"}
	if goruntime.GOOS == "darwin" {
		want = append(want, filepath.Join(root, "wpkd.log")) // systemd logs to the journal instead
	}
	for _, want := range want {
		if !strings.Contains(out.String(), want) {
			t.Errorf("dry run lacks %q:\n%s", want, out.String())
		}
	}
	if len(*calls) != 0 {
		t.Errorf("a dry run ran %q", *calls)
	}
}

func TestServiceStatusAndUninstall(t *testing.T) {
	if goruntime.GOOS != "darwin" && goruntime.GOOS != "linux" {
		t.Skip("no service manager here")
	}
	fakeServices(t)
	var out, errOut bytes.Buffer
	if code := run([]string{"service", "status"}, &out, &errOut); code != 0 || !strings.Contains(out.String(), "not installed") {
		t.Errorf("status: %d %q %q", code, out.String(), errOut.String())
	}
	out.Reset()
	if code := run([]string{"service", "uninstall"}, &out, &errOut); code != 0 || !strings.Contains(out.String(), "wpkd is stopped") {
		t.Errorf("uninstall: %d %q", code, out.String())
	}
	if code := run([]string{"service", "bogus"}, &out, &errOut); code != 2 {
		t.Errorf("bogus subcommand: %d", code)
	}
}

func TestDoctorWithoutDaemon(t *testing.T) {
	t.Setenv("WPK_SOCKET", filepath.Join(t.TempDir(), "none.sock"))
	var out, errOut bytes.Buffer
	if code := run([]string{"doctor"}, &out, &errOut); code != 1 {
		t.Fatalf("exit %d", code)
	}
	if !strings.Contains(out.String(), "FAIL") || !strings.Contains(out.String(), "wpk service install") {
		t.Errorf("doctor output:\n%s", out.String())
	}
}
