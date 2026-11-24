package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

func runSquash(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("squash", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var tags stringList
	fs.Var(&tags, "t", "name and optional tag for the squashed image (repeatable)")
	expert := fs.String("expert", "", "for an image with a MIX, the one expert to bake in")
	to := fs.String("to", "dense", "what to squash into; only dense is supported so far")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk squash [-t name[:tag]]... [--expert name] [--to dense] <ref>\n\nBakes the image's LoRA layers into its base tensors and stores the result as\na new image with a single base layer. The math runs in wpkd's build worker.")
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
	if *to != "dense" {
		fmt.Fprintf(stderr, "wpk squash: --to %s is not supported; only dense is so far\n", *to)
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res, err := client().Squash(ctx, api.SquashRequest{Ref: pos[0], Expert: *expert, Tags: tags}, printEvent(stdout, stderr))
	if err != nil {
		fmt.Fprintf(stderr, "wpk squash: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Squashed into %s: %d blobs, %s (%s new)\n",
		res.Manifest.Digest, res.Blobs, api.FormatSize(res.Size), api.FormatSize(res.NewBytes))
	for _, t := range res.Tags {
		fmt.Fprintf(stdout, "Tagged %s\n", t)
	}
	return 0
}
