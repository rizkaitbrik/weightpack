package service

import (
	"encoding/xml"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type recorder struct {
	calls [][]string
	out   map[string]string // command line -> output
	fail  map[string]bool
}

func (r *recorder) run(name string, args ...string) ([]byte, error) {
	line := strings.Join(append([]string{name}, args...), " ")
	r.calls = append(r.calls, append([]string{name}, args...))
	if r.fail[line] {
		return []byte("boom"), errors.New("exit status 1")
	}
	return []byte(r.out[line]), nil
}

func (r *recorder) lines() []string {
	var out []string
	for _, c := range r.calls {
		out = append(out, strings.Join(c, " "))
	}
	return out
}

var base = Spec{
	Wpkd: "/opt/wpk/bin/wpkd",
	Args: []string{"--worker", "uv run --project /src/my project/python/worker weightpack-worker", "--gateway", "127.0.0.1:8090"},
	Env:  map[string]string{"PATH": "/opt/homebrew/bin:/usr/bin", "HF_HOME": "/data/100%$hf"},
}

// specIn is the test spec with its log under home.
func specIn(home string) Spec {
	s := base
	s.Log = filepath.Join(home, ".weightpack", "wpkd.log")
	return s
}

func TestLaunchdAgent(t *testing.T) {
	home := t.TempDir()
	rec := &recorder{out: map[string]string{
		"launchctl print gui/501/" + Label: "gui/501/dev.weightpack.wpkd = {\n\tstate = running\n\tpid = 4242\n}",
	}}
	m := Manager{OS: "darwin", Home: home, UID: 501, Run: rec.run}
	spec := specIn(home)
	unit, err := m.Render(spec)
	if err != nil {
		t.Fatal(err)
	}
	// The plist is well-formed XML and keeps every argument whole.
	dec := xml.NewDecoder(strings.NewReader(string(unit)))
	dec.Strict = true
	var strs []string
	for {
		tok, err := dec.Token()
		if err != nil {
			break
		}
		if cd, ok := tok.(xml.CharData); ok && strings.TrimSpace(string(cd)) != "" {
			strs = append(strs, string(cd))
		}
	}
	for _, want := range []string{spec.Wpkd, spec.Args[1], "/data/100%$hf", spec.Log, Label} {
		if !slices.Contains(strs, want) {
			t.Errorf("plist lacks %q", want)
		}
	}

	p, err := m.Install(spec)
	if err != nil {
		t.Fatal(err)
	}
	if p != filepath.Join(home, "Library/LaunchAgents/dev.weightpack.wpkd.plist") {
		t.Errorf("path %s", p)
	}
	if got, _ := os.ReadFile(p); string(got) != string(unit) {
		t.Error("installed plist differs from the rendered one")
	}
	want := []string{"launchctl bootout gui/501/" + Label, "launchctl bootstrap gui/501 " + p}
	if !slices.Equal(rec.lines(), want) {
		t.Errorf("commands %q", rec.lines())
	}
	if st, _ := m.Status(); st != "running, pid 4242 ("+p+")" {
		t.Errorf("status %q", st)
	}
	if _, err := m.Uninstall(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p); !os.IsNotExist(err) {
		t.Error("uninstall left the plist")
	}
	if st, _ := m.Status(); st != "not installed" {
		t.Errorf("status after uninstall %q", st)
	}
}

func TestSystemdUnit(t *testing.T) {
	home := t.TempDir()
	rec := &recorder{out: map[string]string{"systemctl --user is-active " + Unit: "active\n"}}
	m := Manager{OS: "linux", Home: home, Run: rec.run}
	spec := specIn(home)
	unit, err := m.Render(spec)
	if err != nil {
		t.Fatal(err)
	}
	text := string(unit)
	for _, want := range []string{
		`ExecStart=/opt/wpk/bin/wpkd --worker "uv run --project /src/my project/python/worker weightpack-worker" --gateway 127.0.0.1:8090`,
		`Environment=HF_HOME=/data/100%%$$hf`,
		`Environment=PATH=/opt/homebrew/bin:/usr/bin`,
		"Restart=on-failure", "WantedBy=default.target",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("unit lacks %q:\n%s", want, text)
		}
	}
	p, err := m.Install(spec)
	if err != nil {
		t.Fatal(err)
	}
	if p != filepath.Join(home, ".config/systemd/user/wpkd.service") {
		t.Errorf("path %s", p)
	}
	want := []string{"systemctl --user daemon-reload", "systemctl --user enable " + Unit, "systemctl --user restart " + Unit}
	if !slices.Equal(rec.lines(), want) {
		t.Errorf("commands %q", rec.lines())
	}
	if st, _ := m.Status(); st != "active ("+p+")" {
		t.Errorf("status %q", st)
	}
	if _, err := m.Uninstall(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p); !os.IsNotExist(err) {
		t.Error("uninstall left the unit")
	}
}

func TestInstallErrors(t *testing.T) {
	rec := &recorder{fail: map[string]bool{"systemctl --user restart " + Unit: true}}
	home := t.TempDir()
	m := Manager{OS: "linux", Home: home, Run: rec.run}
	if _, err := m.Install(specIn(home)); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("failed restart: %v", err)
	}
	if _, err := m.Render(Spec{Wpkd: "wpkd"}); err == nil {
		t.Error("a relative wpkd path was accepted")
	}
	if _, err := (Manager{OS: "windows"}).Path(); !errors.Is(err, ErrUnsupported) {
		t.Errorf("windows: %v", err)
	}
}

func TestSystemdQuote(t *testing.T) {
	for in, want := range map[string]string{
		"plain":      "plain",
		"two words":  `"two words"`,
		`say "hi"`:   `"say \"hi\""`,
		"50%":        "50%%",
		"$HOME":      "$$HOME",
		`back\slash`: `"back\\slash"`,
	} {
		if got := systemdQuote(in); got != want {
			t.Errorf("systemdQuote(%q) = %s, want %s", in, got, want)
		}
	}
}
