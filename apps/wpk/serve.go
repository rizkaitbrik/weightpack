package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

func runRun(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	name := fs.String("name", "", "model name clients send (default: the image reference)")
	engine := fs.String("engine", "", "reference or vllm (default: vllm on NVIDIA GPUs when installed, else reference)")
	device := fs.String("device", "", "cuda, mps or cpu (default: wpkd's, else detected)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk run [--name model] [--engine reference|vllm] [--device cuda|mps|cpu] <ref>\n\n"+
			"Serves an image on wpkd's OpenAI-compatible gateway. Images on the same base\n"+
			"share one engine and one copy of the base weights; each one's top LoRA\n"+
			"layer (or each expert of a MIX) is an adapter in that engine.")
		fs.PrintDefaults()
	}
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		return 2
	}
	if len(pos) != 1 {
		fs.Usage()
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	c, err := client().Run(ctx, api.RunRequest{Ref: pos[0], Name: *name, Engine: *engine, Device: *device}, printEvent(stdout, stderr))
	if err != nil {
		fmt.Fprintf(stderr, "wpk run: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Running %s on the %s engine (%s)\n", c.Name, c.Engine, c.Device)
	if c.Endpoint != "" {
		fmt.Fprintf(stdout, "  endpoint  %s\n", c.Endpoint)
	}
	for _, m := range c.Models {
		if m == c.Name && c.Router != "" {
			fmt.Fprintf(stdout, "  model     %s  (routes each token across the experts: %s)\n", m, c.Router)
			continue
		}
		fmt.Fprintf(stdout, "  model     %s\n", m)
	}
	if c.Endpoint != "" && len(c.Models) > 0 {
		fmt.Fprintf(stdout, "\nTry it:\n  curl %s/chat/completions -H 'Content-Type: application/json' \\\n    -d '{\"model\": %q, \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'\n", c.Endpoint, c.Models[0])
	}
	return 0
}

func runPs(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ps", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk ps\n\nLists running containers and the models they answer to.")
	}
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 {
		return 2
	}
	cs, err := client().Containers(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "wpk ps: %v\n", err)
		return 1
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tIMAGE\tMODELS\tENGINE\tDEVICE\tSTATUS\tUP")
	for _, c := range cs {
		status := c.Status
		if c.Error != "" {
			status += " (" + c.Error + ")"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s %s\t%s\t%s\t%s\n", c.Name, c.Image, strings.Join(c.Models, ","),
			c.Engine, c.EngineID, c.Device, status, time.Since(c.Created).Round(time.Second))
	}
	_ = tw.Flush()
	return 0
}

func runStop(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprintln(stderr, "usage: wpk stop <name>...\n\nStops containers. An engine stops with the last container on it.")
		return 2
	}
	res, err := client().Stop(context.Background(), api.StopRequest{Names: args})
	if err != nil {
		fmt.Fprintf(stderr, "wpk stop: %v\n", err)
		return 1
	}
	for _, n := range res.Stopped {
		fmt.Fprintf(stdout, "Stopped %s\n", n)
	}
	return 0
}

func runLogs(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("logs", flag.ContinueOnError)
	fs.SetOutput(stderr)
	follow := fs.Bool("f", false, "keep printing new output")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk logs [-f] <name>\n\nPrints the log of the engine serving a container (containers on one engine\nshare it).")
		fs.PrintDefaults()
	}
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		return 2
	}
	if len(pos) != 1 {
		fs.Usage()
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	rc, err := client().Logs(ctx, pos[0], *follow)
	if err != nil {
		fmt.Fprintf(stderr, "wpk logs: %v\n", err)
		return 1
	}
	defer rc.Close()
	if _, err := io.Copy(stdout, rc); err != nil && ctx.Err() == nil {
		fmt.Fprintf(stderr, "wpk logs: %v\n", err)
		return 1
	}
	return 0
}
