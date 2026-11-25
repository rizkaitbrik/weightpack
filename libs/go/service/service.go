// Package service installs wpkd as a per-user login service: a launchd
// agent on macOS, a systemd user unit on Linux. The service starts wpkd at
// login and restarts it if it dies.
package service

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Label names the launchd agent; Unit names the systemd unit.
const (
	Label = "dev.weightpack.wpkd"
	Unit  = "wpkd.service"
)

// Spec is the wpkd command a service runs.
type Spec struct {
	Wpkd string            // absolute path of the wpkd binary
	Args []string          // its flags
	Env  map[string]string // environment, such as PATH
	Log  string            // launchd's log file; systemd logs to the journal
}

// Manager installs services for one user on one OS.
type Manager struct {
	OS   string // "darwin" or "linux"
	Home string
	UID  int
	// Run runs a service manager command (launchctl, systemctl) and returns
	// its combined output.
	Run func(name string, args ...string) ([]byte, error)
}

// ErrUnsupported is returned on systems with neither launchd nor systemd.
var ErrUnsupported = errors.New("wpk service supports macOS (launchd) and Linux (systemd user units)")

// Path is where the service file goes.
func (m Manager) Path() (string, error) {
	switch m.OS {
	case "darwin":
		return filepath.Join(m.Home, "Library", "LaunchAgents", Label+".plist"), nil
	case "linux":
		return filepath.Join(m.Home, ".config", "systemd", "user", Unit), nil
	}
	return "", ErrUnsupported
}

// Render returns the service file for s.
func (m Manager) Render(s Spec) ([]byte, error) {
	if !filepath.IsAbs(s.Wpkd) {
		return nil, fmt.Errorf("wpkd path %q is not absolute", s.Wpkd)
	}
	switch m.OS {
	case "darwin":
		return plist(s), nil
	case "linux":
		return systemdUnit(s), nil
	}
	return nil, ErrUnsupported
}

func xmlText(s string) string {
	var b bytes.Buffer
	_ = xml.EscapeText(&b, []byte(s))
	return b.String()
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func plist(s Spec) []byte {
	var b strings.Builder
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>` + Label + `</string>
  <key>ProgramArguments</key>
  <array>
`)
	for _, a := range append([]string{s.Wpkd}, s.Args...) {
		fmt.Fprintf(&b, "    <string>%s</string>\n", xmlText(a))
	}
	b.WriteString("  </array>\n")
	if len(s.Env) > 0 {
		b.WriteString("  <key>EnvironmentVariables</key>\n  <dict>\n")
		for _, k := range sortedKeys(s.Env) {
			fmt.Fprintf(&b, "    <key>%s</key>\n    <string>%s</string>\n", xmlText(k), xmlText(s.Env[k]))
		}
		b.WriteString("  </dict>\n")
	}
	b.WriteString("  <key>RunAtLoad</key>\n  <true/>\n  <key>KeepAlive</key>\n  <true/>\n")
	// Leave engines time to stop when launchd stops wpkd.
	b.WriteString("  <key>ExitTimeOut</key>\n  <integer>30</integer>\n")
	if s.Log != "" {
		fmt.Fprintf(&b, "  <key>StandardOutPath</key>\n  <string>%s</string>\n", xmlText(s.Log))
		fmt.Fprintf(&b, "  <key>StandardErrorPath</key>\n  <string>%s</string>\n", xmlText(s.Log))
	}
	b.WriteString("</dict>\n</plist>\n")
	return []byte(b.String())
}

// systemdQuote quotes one word of a systemd command line or assignment:
// systemd splits on spaces, expands % specifiers and $ variables, and
// reads C-style escapes inside double quotes.
func systemdQuote(s string) string {
	s = strings.ReplaceAll(s, "%", "%%")
	s = strings.ReplaceAll(s, "$", "$$")
	if s != "" && !strings.ContainsAny(s, " \t\n\"'\\;") {
		return s
	}
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`, "\t", `\t`)
	return `"` + r.Replace(s) + `"`
}

func systemdUnit(s Spec) []byte {
	var b strings.Builder
	b.WriteString("[Unit]\nDescription=Weightpack daemon (wpkd)\nAfter=network-online.target\n\n[Service]\n")
	words := []string{systemdQuote(s.Wpkd)}
	for _, a := range s.Args {
		words = append(words, systemdQuote(a))
	}
	fmt.Fprintf(&b, "ExecStart=%s\n", strings.Join(words, " "))
	for _, k := range sortedKeys(s.Env) {
		fmt.Fprintf(&b, "Environment=%s\n", systemdQuote(k+"="+s.Env[k]))
	}
	b.WriteString("Restart=on-failure\nRestartSec=2\n# Leave engines time to stop.\nTimeoutStopSec=30\n\n[Install]\nWantedBy=default.target\n")
	return []byte(b.String())
}

// Install writes the service file and (re)starts the service.
func (m Manager) Install(s Spec) (string, error) {
	p, err := m.Path()
	if err != nil {
		return "", err
	}
	unit, err := m.Render(s)
	if err != nil {
		return "", err
	}
	if s.Log != "" {
		if err := os.MkdirAll(filepath.Dir(s.Log), 0o755); err != nil {
			return "", err
		}
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(p, unit, 0o644); err != nil {
		return "", err
	}
	switch m.OS {
	case "darwin":
		// Replace a loaded agent, so a reinstall takes the new arguments.
		_, _ = m.Run("launchctl", "bootout", m.domain()+"/"+Label)
		if out, err := m.Run("launchctl", "bootstrap", m.domain(), p); err != nil {
			return p, fmt.Errorf("launchctl bootstrap: %v: %s", err, strings.TrimSpace(string(out)))
		}
	case "linux":
		for _, args := range [][]string{{"--user", "daemon-reload"}, {"--user", "enable", Unit}, {"--user", "restart", Unit}} {
			if out, err := m.Run("systemctl", args...); err != nil {
				return p, fmt.Errorf("systemctl %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
			}
		}
	}
	return p, nil
}

// Uninstall stops the service and removes its file.
func (m Manager) Uninstall() (string, error) {
	p, err := m.Path()
	if err != nil {
		return "", err
	}
	switch m.OS {
	case "darwin":
		_, _ = m.Run("launchctl", "bootout", m.domain()+"/"+Label)
	case "linux":
		_, _ = m.Run("systemctl", "--user", "disable", "--now", Unit)
	}
	if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
		return p, err
	}
	if m.OS == "linux" {
		_, _ = m.Run("systemctl", "--user", "daemon-reload")
	}
	return p, nil
}

var (
	launchdState = regexp.MustCompile(`(?m)^\s*state = (\S+)`)
	launchdPID   = regexp.MustCompile(`(?m)^\s*pid = (\d+)`)
)

// Status says whether the service is installed and running.
func (m Manager) Status() (string, error) {
	p, err := m.Path()
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
		return "not installed", nil
	}
	switch m.OS {
	case "darwin":
		out, err := m.Run("launchctl", "print", m.domain()+"/"+Label)
		if err != nil {
			return "installed, not loaded (" + p + ")", nil
		}
		state := "unknown"
		if s := launchdState.FindSubmatch(out); s != nil {
			state = string(s[1])
		}
		if pid := launchdPID.FindSubmatch(out); pid != nil {
			state += ", pid " + string(pid[1])
		}
		return state + " (" + p + ")", nil
	default:
		out, _ := m.Run("systemctl", "--user", "is-active", Unit)
		return strings.TrimSpace(string(out)) + " (" + p + ")", nil
	}
}

func (m Manager) domain() string { return "gui/" + strconv.Itoa(m.UID) }
