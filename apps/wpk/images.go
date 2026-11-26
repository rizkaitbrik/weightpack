package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
)

// client returns a client for the local daemon ($WPK_SOCKET, or
// $WPK_ROOT/wpkd.sock, or ~/.weightpack/wpkd.sock).
func client() *api.Client { return api.NewClient(api.DefaultSocket()) }

type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

// parseInterleaved parses flags that may come before or after positional
// arguments, as `wpk build . -t name` expects.
func parseInterleaved(fs *flag.FlagSet, args []string) ([]string, error) {
	var positional []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		if fs.NArg() == 0 {
			return positional, nil
		}
		positional = append(positional, fs.Arg(0))
		args = fs.Args()[1:]
	}
}

func runBuild(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("build", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var tags stringList
	fs.Var(&tags, "t", "name and optional tag, such as assistant:v1 (repeatable)")
	noCache := fs.Bool("no-cache", false, "read and hash every source again")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk build [-t name[:tag]]... [directory | Weightfile]\n\nBuilds the Weightfile in the directory (default .). Local sources resolve\nagainst the Weightfile's directory.")
		fs.PrintDefaults()
	}
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		return 2
	}
	if len(pos) > 1 {
		fs.Usage()
		return 2
	}
	path := "."
	if len(pos) == 1 {
		path = pos[0]
	}
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		path = filepath.Join(path, "Weightfile")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		fmt.Fprintf(stderr, "wpk build: %v\n", err)
		return 1
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res, err := client().Build(ctx, api.BuildRequest{Weightfile: abs, Tags: tags, NoCache: *noCache}, printEvent(stdout, stderr))
	if err != nil {
		fmt.Fprintf(stderr, "wpk build: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Built %s: %d blobs, %s (%s new)\n",
		res.Manifest.Digest, res.Blobs, api.FormatSize(res.Size), api.FormatSize(res.NewBytes))
	for _, t := range res.Tags {
		fmt.Fprintf(stdout, "Tagged %s\n", t)
	}
	return 0
}

func runImages(args []string, stdout, stderr io.Writer) int {
	if len(args) != 0 {
		fmt.Fprintln(stderr, "usage: wpk images")
		return 2
	}
	images, err := client().Images(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "wpk images: %v\n", err)
		return 1
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "REF\tDIGEST\tARCH\tLAYERS\tSIZE")
	for _, img := range images {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\n", img.Ref, shortDigest(img.Digest), img.Arch, img.Layers, api.FormatSize(img.Size))
	}
	_ = tw.Flush()
	return 0
}

func runInspect(args []string, stdout, stderr io.Writer) int {
	if len(args) != 1 {
		fmt.Fprintln(stderr, "usage: wpk inspect <ref>")
		return 2
	}
	d, err := client().Inspect(context.Background(), args[0])
	if err != nil {
		fmt.Fprintf(stderr, "wpk inspect: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(d)
	return 0
}

func runHistory(args []string, stdout, stderr io.Writer) int {
	if len(args) != 1 {
		fmt.Fprintln(stderr, "usage: wpk history <ref>")
		return 2
	}
	d, err := client().Inspect(context.Background(), args[0])
	if err != nil {
		fmt.Fprintf(stderr, "wpk history: %v\n", err)
		return 1
	}
	type stats struct {
		blobs int
		size  int64
	}
	per := map[string]*stats{}
	for _, b := range d.Manifest.Layers {
		id := b.Annotations[spec.AnnotationLayer]
		if per[id] == nil {
			per[id] = &stats{}
		}
		per[id].blobs++
		per[id].size += b.Size
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LAYER\tKIND\tPARENT\tDETAIL\tBLOBS\tSIZE\tCHAIN")
	row := func(id, kind, parent, detail, chain string) {
		st := per[id]
		if st == nil {
			st = &stats{}
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%s\t%s\n", id, kind, orDash(parent), detail, st.blobs, api.FormatSize(st.size), shortDigest(chain))
	}
	for _, l := range d.Config.Layers {
		row(l.ID, string(l.Kind), l.Parent, layerDetail(l, d.Config.Arch), l.Chain)
	}
	if per[""] != nil {
		row("", "meta", "", "image-level files", "")
	}
	_ = tw.Flush()
	return 0
}

func layerDetail(l spec.Layer, arch spec.Arch) string {
	switch l.Kind {
	case spec.KindBase:
		return fmt.Sprintf("%s, %d blocks, hidden %d", orDash(arch.Family), arch.Blocks, arch.Hidden)
	case spec.KindLoRA:
		scale := ""
		if l.RSLoRA {
			scale = ", rslora"
		}
		return fmt.Sprintf("rank %d, alpha %s%s, %s", l.Rank, strconv.FormatFloat(l.Alpha, 'f', -1, 64), scale, strings.Join(l.Targets, ","))
	case spec.KindMix:
		blocks := l.Blocks
		if blocks == "" {
			blocks = "all"
		}
		return fmt.Sprintf("top_k %d of %s, router %s, blocks %s", l.TopK, strings.Join(l.Experts, ","), l.Router, blocks)
	}
	return ""
}

func shortDigest(d string) string {
	if h, ok := strings.CutPrefix(d, "sha256:"); ok && len(h) >= 12 {
		return h[:12]
	}
	return orDash(d)
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
