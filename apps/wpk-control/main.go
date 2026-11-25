// Command wpk-control is Weightpack's cluster control plane: an
// OpenAI-compatible gateway in front of the node agents (wpkd --api on each
// GPU node) that keeps the catalogue's images running on them, placing
// each on a node that already holds its base. See libs/go/control.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/control"
)

var version = "dev"

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return def
}

func main() {
	listen := flag.String("listen", envOr("WPK_CONTROL_LISTEN", ":8080"), "address of the OpenAI gateway and the status API ($WPK_CONTROL_LISTEN)")
	agents := flag.String("agents", os.Getenv("WPK_AGENTS"), "node agents: dns://<headless service>:<port>, or comma-separated http://<host>:<port> ($WPK_AGENTS)")
	catalogue := flag.String("catalogue", envOr("WPK_CATALOGUE", "/etc/weightpack/catalogue.json"), "catalogue of the models to serve ($WPK_CATALOGUE)")
	interval := flag.Duration("interval", 5*time.Second, "time between passes over the agents")
	showVersion := flag.Bool("version", false, "print the version and exit")
	flag.Parse()
	if *showVersion {
		fmt.Println("wpk-control", version)
		return
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := run(*listen, *agents, *catalogue, *interval, logger); err != nil {
		logger.Error("wpk-control stopped", "err", err)
		os.Exit(1)
	}
}

// run serves until SIGINT or SIGTERM. The keys come from the environment,
// as a Kubernetes Secret provides them: $WPK_AGENT_KEY for the agents'
// API and gateways, $WPK_API_KEY for clients.
func run(listen, agents, catalogue string, interval time.Duration, logger *slog.Logger) error {
	d, err := control.ParseAgents(agents)
	if err != nil {
		return err
	}
	if os.Getenv("WPK_AGENT_KEY") == "" {
		return errors.New("set $WPK_AGENT_KEY to the agents' API key (wpkd --api-key)")
	}
	c := control.New(control.Config{
		Discovery: d, AgentKey: os.Getenv("WPK_AGENT_KEY"), APIKey: os.Getenv("WPK_API_KEY"),
		Catalogue: catalogue, Interval: interval, Logger: logger,
	})
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go c.Run(ctx)

	srv := &http.Server{Addr: listen, Handler: c, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()
	logger.Info("wpk-control listening", "addr", listen, "agents", agents, "catalogue", catalogue, "version", version)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
