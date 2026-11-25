package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/service"
)

// serviceManager is the service manager for this user; tests replace it.
var serviceManager = func() (service.Manager, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return service.Manager{}, err
	}
	return service.Manager{
		OS: goruntime.GOOS, Home: home, UID: os.Getuid(),
		Run: func(name string, args ...string) ([]byte, error) {
			return exec.Command(name, args...).CombinedOutput()
		},
	}, nil
}

// serviceEnv lists the variables a service keeps from the installing shell:
// launchd and systemd start services with almost none, so without PATH
// wpkd could not find uv, the worker, the engines or nvidia-smi. Tokens
// stay out of service files; hf auth login saves one under HF_HOME.
var serviceEnv = []string{"PATH", "HF_HOME", "HF_ENDPOINT", "CUDA_VISIBLE_DEVICES", "WPK_ROOT"}

func runService(args []string, stdout, stderr io.Writer) int {
	usage := func() {
		fmt.Fprintln(stderr, "usage: wpk service install [flags] | uninstall | status\n\n"+
			"Runs wpkd as a login service (a launchd agent on macOS, a systemd user unit\n"+
			"on Linux) that starts at login and restarts if it dies.")
	}
	if len(args) == 0 {
		usage()
		return 2
	}
	m, err := serviceManager()
	if err != nil {
		fmt.Fprintf(stderr, "wpk service: %v\n", err)
		return 1
	}
	switch args[0] {
	case "install":
		return serviceInstall(m, args[1:], stdout, stderr)
	case "uninstall":
		p, err := m.Uninstall()
		if err != nil {
			fmt.Fprintf(stderr, "wpk service uninstall: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "Removed %s; wpkd is stopped. The store is untouched.\n", p)
		return 0
	case "status":
		st, err := m.Status()
		if err != nil {
			fmt.Fprintf(stderr, "wpk service status: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "wpkd service: %s\n", st)
		return 0
	}
	usage()
	return 2
}

func serviceInstall(m service.Manager, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("service install", flag.ContinueOnError)
	fs.SetOutput(stderr)
	wpkd := fs.String("wpkd", "", "path of the wpkd binary (default: next to wpk, else on PATH)")
	workerCmd := fs.String("worker", "", "command that starts the build worker (default: weightpack-worker, if on PATH)")
	refCmd := fs.String("engine-reference", "", "command that starts the reference engine (default: wpkd's, weightpack-engine)")
	vllmCmd := fs.String("engine-vllm", "", "command that starts vLLM (default: wpkd's, vllm serve)")
	gateway := fs.String("gateway", "", "gateway address (default: wpkd's, 127.0.0.1:8090)")
	gatewayKey := fs.String("gateway-key", "", "bearer token the gateway requires (stored in the service file)")
	root := fs.String("root", "", "store directory (default: $WPK_ROOT, else ~/.weightpack)")
	dry := fs.Bool("dry-run", false, "print the service file instead of installing it")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk service install [flags]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 {
		return 2
	}
	fail := func(format string, a ...any) int {
		fmt.Fprintf(stderr, "wpk service install: "+format+"\n", a...)
		return 1
	}

	bin := *wpkd
	if bin == "" {
		if self, err := os.Executable(); err == nil {
			if p := filepath.Join(filepath.Dir(self), "wpkd"); isFile(p) {
				bin = p
			}
		}
	}
	if bin == "" {
		if p, err := exec.LookPath("wpkd"); err == nil {
			bin = p
		}
	}
	if bin == "" {
		return fail("wpkd not found next to wpk or on PATH; pass --wpkd")
	}
	bin, err := filepath.Abs(bin)
	if err != nil {
		return fail("%v", err)
	}
	if *workerCmd == "" {
		if _, err := exec.LookPath("weightpack-worker"); err != nil {
			return fail("no build worker on PATH: pip install weightpack-worker, or pass --worker \"<command>\"")
		}
		*workerCmd = "weightpack-worker"
	}
	if *refCmd == "" {
		if _, err := exec.LookPath("weightpack-engine"); err != nil {
			fmt.Fprintln(stderr, `warning: weightpack-engine is not on PATH, so wpk run will fail until it is: pip install "weightpack-worker[serve]", or pass --engine-reference`)
		}
	}
	if *root == "" {
		*root = api.DefaultRoot()
	}
	absRoot, err := filepath.Abs(*root)
	if err != nil {
		return fail("%v", err)
	}

	spec := service.Spec{Wpkd: bin, Args: []string{"--root", absRoot, "--worker", *workerCmd}, Env: map[string]string{}, Log: filepath.Join(absRoot, "wpkd.log")}
	for flagName, v := range map[string]string{"--engine-reference": *refCmd, "--engine-vllm": *vllmCmd, "--gateway": *gateway, "--gateway-key": *gatewayKey} {
		if v != "" {
			spec.Args = append(spec.Args, flagName, v)
		}
	}
	for _, k := range serviceEnv {
		if v, ok := os.LookupEnv(k); ok && k != "WPK_ROOT" {
			spec.Env[k] = v
		}
	}

	if *dry {
		unit, err := m.Render(spec)
		if err != nil {
			return fail("%v", err)
		}
		p, _ := m.Path()
		fmt.Fprintf(stdout, "# %s\n%s", p, unit)
		return 0
	}
	p, err := m.Install(spec)
	if err != nil {
		return fail("%v", err)
	}
	fmt.Fprintf(stdout, "Installed %s\n", p)

	// Wait for the daemon to answer on its socket.
	socket := filepath.Join(absRoot, "wpkd.sock")
	c := api.NewClient(socket)
	deadline := time.Now().Add(20 * time.Second)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		h, err := c.Health(ctx)
		cancel()
		if err == nil {
			fmt.Fprintf(stdout, "wpkd %s is running (store %s)\n", h["version"], absRoot)
			break
		}
		if time.Now().After(deadline) {
			fmt.Fprintf(stderr, "wpkd did not answer on %s yet; its log is %s\n", socket, spec.Log)
			return 1
		}
		time.Sleep(250 * time.Millisecond)
	}
	if absRoot != mustAbs(api.DefaultRoot()) {
		fmt.Fprintf(stdout, "Set WPK_ROOT=%s so wpk finds this daemon.\n", absRoot)
	}
	if m.OS == "linux" {
		fmt.Fprintln(stdout, "To keep wpkd running after you log out: loginctl enable-linger "+os.Getenv("USER"))
	}
	fmt.Fprintln(stdout, "Check the machine with: wpk doctor")
	return 0
}

func isFile(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.Mode().IsRegular()
}

func mustAbs(p string) string {
	if a, err := filepath.Abs(p); err == nil {
		return a
	}
	return p
}
