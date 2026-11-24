// Command wpkd is the Weightpack daemon. It owns the local blob store, runs
// builds and, on GPU nodes, acts as the node agent. It serves its API (see
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
	"syscall"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/daemon"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

var version = "dev"

func main() {
	root := flag.String("root", api.DefaultRoot(), "directory of the local store ($WPK_ROOT)")
	socket := flag.String("socket", "", "Unix socket to listen on (default $WPK_SOCKET, else <root>/wpkd.sock)")
	flag.Parse()
	if *socket == "" {
		*socket = os.Getenv("WPK_SOCKET")
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := serve(ctx, *root, *socket, logger); err != nil {
		logger.Error("wpkd stopped", "err", err)
		os.Exit(1)
	}
}

// serve opens the store and serves the API until ctx is done.
func serve(ctx context.Context, root, socket string, logger *slog.Logger) error {
	s, err := store.Open(root)
	if err != nil {
		return err
	}
	if socket == "" {
		socket = filepath.Join(root, "wpkd.sock")
	}
	_ = os.Remove(socket)
	ln, err := net.Listen("unix", socket)
	if err != nil {
		return err
	}
	if err := os.Chmod(socket, 0o600); err != nil {
		_ = ln.Close()
		return err
	}

	srv := &http.Server{Handler: daemon.NewHandler(s, version, logger), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()

	logger.Info("wpkd listening", "socket", socket, "root", root, "version", version)
	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
