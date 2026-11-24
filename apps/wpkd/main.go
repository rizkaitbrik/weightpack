// Command wpkd is the Weightpack daemon. It owns the local blob store and,
// on GPU nodes, acts as the node agent. This skeleton opens the store and
// serves a health endpoint on a Unix socket.
package main

import (
	"context"
	"encoding/json"
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

	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

var version = "dev"

func main() {
	root := flag.String("root", defaultRoot(), "directory of the local store")
	socket := flag.String("socket", "", "Unix socket to listen on (default <root>/wpkd.sock)")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := serve(*root, *socket, logger); err != nil {
		logger.Error("wpkd stopped", "err", err)
		os.Exit(1)
	}
}

func defaultRoot() string {
	if dir := os.Getenv("WPK_ROOT"); dir != "" {
		return dir
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".weightpack")
	}
	return ".weightpack"
}

func serve(root, socket string, logger *slog.Logger) error {
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

	srv := &http.Server{Handler: newMux(s), ReadHeaderTimeout: 5 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
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

func newMux(s *store.Store) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{
			"status":  "ok",
			"version": version,
			"root":    s.Root(),
		})
	})
	mux.HandleFunc("GET /v1/blobs/{digest}", func(w http.ResponseWriter, r *http.Request) {
		ok, err := s.Has(r.PathValue("digest"))
		switch {
		case err != nil:
			http.Error(w, err.Error(), http.StatusBadRequest)
		case !ok:
			http.Error(w, "not found", http.StatusNotFound)
		default:
			w.WriteHeader(http.StatusNoContent)
		}
	})
	return mux
}
