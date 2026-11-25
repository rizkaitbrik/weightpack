// Command wpkd is the Weightpack daemon. It owns the local blob store, runs
// builds, pushes and pulls, hands per-weight work (squash) to a Python build
// worker, and runs containers: engine processes that serve images, behind an
// OpenAI-compatible gateway. It serves its API (see libs/go/api) on a Unix
// socket that only its owner can connect to, and, for a cluster control
// plane, on a TCP address with a bearer key (--api).
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
	"strconv"
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
	api          string // TCP address of the API for a control plane; "" for none
	apiToken     string // bearer token the TCP API requires
	policyFile   string // trust policy file; "" means <root>/policy.json
	maxEngines   int
	node         string // machine name /v1/node reports; "" means the hostname
	insecure     string // comma-separated registries reached over plain HTTP
}

// envInt returns $key as a number, or 0.
func envInt(key string) int {
	n, _ := strconv.Atoi(os.Getenv(key))
	return n
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
	flag.StringVar(&c.api, "api", os.Getenv("WPK_API"), "also serve the API on this TCP address, for a cluster control plane; needs --api-key ($WPK_API)")
	flag.StringVar(&c.apiToken, "api-key", os.Getenv("WPK_API_KEY"), "bearer token the TCP API requires ($WPK_API_KEY)")
	flag.StringVar(&c.policyFile, "trust-policy", os.Getenv("WPK_TRUST_POLICY"), "trust policy file, such as a mounted ConfigMap ($WPK_TRUST_POLICY, default <root>/policy.json)")
	flag.IntVar(&c.maxEngines, "max-engines", envInt("WPK_MAX_ENGINES"), "most engine processes to run at once, 0 for no limit ($WPK_MAX_ENGINES)")
	flag.StringVar(&c.node, "node-name", os.Getenv("WPK_NODE_NAME"), "this machine's name for a control plane ($WPK_NODE_NAME, default the hostname)")
	flag.StringVar(&c.insecure, "insecure-registries", os.Getenv("WPK_INSECURE_REGISTRIES"), "comma-separated registries (host[:port]) to reach over plain HTTP, as Docker's insecure-registries ($WPK_INSECURE_REGISTRIES)")
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

	if c.api != "" && c.apiToken == "" {
		_ = ln.Close()
		return errors.New("--api serves the whole API over TCP; set --api-key (or $WPK_API_KEY) too")
	}
	opts := []daemon.Option{daemon.WithWorker(wc), daemon.WithHF(hf.NewFromEnv(c.hfCache))}
	if c.policyFile != "" {
		opts = append(opts, daemon.WithPolicyFile(c.policyFile))
	}
	var insecure []string
	for _, h := range strings.Split(c.insecure, ",") {
		if h = strings.TrimSpace(h); h != "" {
			insecure = append(insecure, h)
		}
	}
	opts = append(opts, daemon.WithInsecureRegistries(insecure))
	gatewayPort := 0
	var gw *http.Server
	if c.gateway != "" {
		gln, err := net.Listen("tcp", c.gateway)
		if err != nil {
			_ = ln.Close()
			return fmt.Errorf("gateway: %w (pick another address with --gateway, or turn serving off with --gateway '')", err)
		}
		gatewayPort = gln.Addr().(*net.TCPAddr).Port
		rt := serving.New(serving.Config{
			Store: s, Worker: wc, Engines: c.engines, Device: c.device, MaxEngines: c.maxEngines,
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

	opts = append(opts, daemon.WithNode(c.node, gatewayPort))
	handler := daemon.NewHandler(s, version, logger, opts...)
	srv := &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second}
	var apiSrv *http.Server
	if c.api != "" {
		aln, err := net.Listen("tcp", c.api)
		if err != nil {
			_ = ln.Close()
			return fmt.Errorf("api: %w", err)
		}
		apiSrv = &http.Server{Handler: daemon.RequireKey(handler, c.apiToken), ReadHeaderTimeout: 5 * time.Second}
		go func() {
			if err := apiSrv.Serve(aln); err != nil && !errors.Is(err, http.ErrServerClosed) {
				logger.Error("api listener stopped", "err", err)
			}
		}()
		logger.Info("api listening", "addr", aln.Addr().String())
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
		if gw != nil {
			_ = gw.Shutdown(shutdown)
		}
		if apiSrv != nil {
			_ = apiSrv.Shutdown(shutdown)
		}
	}()

	logger.Info("wpkd listening", "socket", c.socket, "root", c.root, "worker", c.workerSocket, "version", version)
	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
