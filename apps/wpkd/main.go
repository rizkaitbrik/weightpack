// Command wpkd is the Weightpack daemon. It owns the local blob store, runs
// builds, pushes and pulls, hands per-weight work (squash) to a Python build
// worker, and runs containers: engine processes that serve images, behind an
// OpenAI-compatible gateway. It serves its API (see libs/go/api) on a Unix
// socket that only its owner can connect to.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
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
	"github.com/rizkaitbrik/weightpack/libs/go/serving"
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
	gateway      string   // TCP address of the OpenAI gateway; "" turns serving off
	device       string   // default device for containers; "" detects
	engines      map[string][]string
	apiKey       string // bearer token the gateway requires; "" for none
}

// envOr returns $key, or def when it is unset.
func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return def
}

func main() {
	var c config
	flag.StringVar(&c.root, "root", api.DefaultRoot(), "directory of the local store ($WPK_ROOT)")
	flag.StringVar(&c.socket, "socket", os.Getenv("WPK_SOCKET"), "Unix socket to listen on ($WPK_SOCKET, default <root>/wpkd.sock)")
	flag.StringVar(&c.workerSocket, "worker-socket", "", "socket of the build worker ($WPK_WORKER_SOCKET, default <root>/worker.sock)")
	workerCmd := flag.String("worker", os.Getenv("WPK_WORKER"), "command that starts a build worker, such as weightpack-worker; wpkd adds --socket and stops it on exit ($WPK_WORKER)")
	flag.StringVar(&c.hfCache, "hf-cache", os.Getenv("WPK_HF_CACHE"), "where hf:// downloads are cached ($WPK_HF_CACHE, default <root>/cache/hf)")
	flag.StringVar(&c.gateway, "gateway", envOr("WPK_GATEWAY", "127.0.0.1:8090"), `address of the OpenAI-compatible gateway containers answer on; "" turns serving off ($WPK_GATEWAY)`)
	flag.StringVar(&c.device, "device", os.Getenv("WPK_DEVICE"), "default device for containers: cuda, mps or cpu ($WPK_DEVICE, default: detect)")
	flag.StringVar(&c.apiKey, "gateway-key", os.Getenv("WPK_GATEWAY_KEY"), "bearer token the gateway requires; set it when the gateway listens beyond loopback ($WPK_GATEWAY_KEY)")
	refCmd := flag.String("engine-reference", envOr("WPK_ENGINE_REFERENCE", "weightpack-engine"), "command that starts the reference engine ($WPK_ENGINE_REFERENCE)")
	vllmCmd := flag.String("engine-vllm", envOr("WPK_ENGINE_VLLM", "vllm serve"), "command that starts vLLM ($WPK_ENGINE_VLLM)")
	flag.Parse()
	c.workerCmd = strings.Fields(*workerCmd)
	c.engines = map[string][]string{serving.EngineReference: strings.Fields(*refCmd), serving.EngineVLLM: strings.Fields(*vllmCmd)}

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

	opts := []daemon.Option{daemon.WithWorker(wc), daemon.WithHF(hf.NewFromEnv(c.hfCache))}
	var gw *http.Server
	if c.gateway != "" {
		gln, err := net.Listen("tcp", c.gateway)
		if err != nil {
			_ = ln.Close()
			return fmt.Errorf("gateway: %w (pick another address with --gateway, or turn serving off with --gateway '')", err)
		}
		rt := serving.New(serving.Config{
			Store: s, Worker: wc, Engines: c.engines, Device: c.device,
			Endpoint: "http://" + gln.Addr().String() + "/v1", Logger: logger, APIKey: c.apiKey,
		})
		defer rt.Close()
		opts = append(opts, daemon.WithRuntime(rt))
		gw = &http.Server{Handler: rt.Gateway(), ReadHeaderTimeout: 10 * time.Second}
		go func() {
			if err := gw.Serve(gln); err != nil && !errors.Is(err, http.ErrServerClosed) {
				logger.Error("gateway stopped", "err", err)
			}
		}()
		logger.Info("gateway listening", "addr", gln.Addr().String())
	}

	srv := &http.Server{
		Handler:           daemon.NewHandler(s, version, logger, opts...),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
		if gw != nil {
			_ = gw.Shutdown(shutdown)
		}
	}()

	logger.Info("wpkd listening", "socket", c.socket, "root", c.root, "worker", c.workerSocket, "version", version)
	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
