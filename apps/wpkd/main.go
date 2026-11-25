// Command wpkd is the Weightpack daemon. It owns the local blob store, runs
// builds, pushes and pulls, hands per-weight work (squash) to a Python build
// worker and, on GPU nodes, acts as the node agent. It serves its API (see
// libs/go/api) on a Unix socket that only its owner can connect to.
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/daemon"
	"github.com/rizkaitbrik/weightpack/libs/go/hf"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/worker"
)

var version = "dev"

// config is what wpkd serves: its store, its socket and its build worker.
type config struct {
	root         string
	socket       string   // default <root>/wpkd.sock
	workerSocket string   // default <root>/worker.sock
	workerCmd    []string // start this worker on workerSocket; empty: connect only
	hfCache      string   // default <root>/cache/hf
}

func main() {
	var c config
	flag.StringVar(&c.root, "root", api.DefaultRoot(), "directory of the local store ($WPK_ROOT)")
	flag.StringVar(&c.socket, "socket", os.Getenv("WPK_SOCKET"), "Unix socket to listen on ($WPK_SOCKET, default <root>/wpkd.sock)")
	flag.StringVar(&c.workerSocket, "worker-socket", "", "socket of the build worker ($WPK_WORKER_SOCKET, default <root>/worker.sock)")
	workerCmd := flag.String("worker", os.Getenv("WPK_WORKER"), "command that starts a build worker, such as weightpack-worker; wpkd adds --socket and stops it on exit ($WPK_WORKER)")
	flag.StringVar(&c.hfCache, "hf-cache", os.Getenv("WPK_HF_CACHE"), "where hf:// downloads are cached ($WPK_HF_CACHE, default <root>/cache/hf)")
	flag.Parse()
	c.workerCmd = strings.Fields(*workerCmd)

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := serve(ctx, c, logger); err != nil {
		logger.Error("wpkd stopped", "err", err)
		os.Exit(1)
	}
}

// serve opens the store, connects to (or starts) the build worker and serves
// the API until ctx is done.
func serve(ctx context.Context, c config, logger *slog.Logger) error {
	s, err := store.Open(c.root)
	if err != nil {
		return err
	}
	if c.socket == "" {
		c.socket = filepath.Join(c.root, "wpkd.sock")
	}
	if c.workerSocket == "" {
		c.workerSocket = worker.DefaultSocket(c.root)
	}
	if c.hfCache == "" {
		c.hfCache = filepath.Join(c.root, "cache", "hf")
	}
	if len(c.workerCmd) > 0 {
		logger.Info("starting build worker", "command", strings.Join(c.workerCmd, " "), "socket", c.workerSocket)
		p, err := worker.Spawn(ctx, c.workerCmd, c.workerSocket, os.Stderr, time.Minute)
		if err != nil {
			return err
		}
		defer p.Stop()
	}
	wc, err := worker.Dial(c.workerSocket)
	if err != nil {
		return err
	}
	defer wc.Close()

	_ = os.Remove(c.socket)
	ln, err := net.Listen("unix", c.socket)
	if err != nil {
		return err
	}
	if err := os.Chmod(c.socket, 0o600); err != nil {
		_ = ln.Close()
		return err
	}

	srv := &http.Server{
		Handler:           daemon.NewHandler(s, version, logger, daemon.WithWorker(wc), daemon.WithHF(hf.NewFromEnv(c.hfCache))),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()

	logger.Info("wpkd listening", "socket", c.socket, "root", c.root, "worker", c.workerSocket, "version", version)
	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
