package main

import (
	"context"
	"flag"
	"fmt"
	"io"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

func runRm(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: wpk rm <ref>...\n\nUntags images. Their blobs stay until wpk prune.")
		return 2
	}
	res, err := client().Rm(context.Background(), api.RmRequest{Refs: args})
	if err != nil {
		fmt.Fprintf(stderr, "wpk rm: %v\n", err)
		return 1
	}
	for _, r := range res.Removed {
		fmt.Fprintf(stdout, "Untagged %s\n", r)
	}
	return 0
}

func runPrune(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("prune", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dry := fs.Bool("dry-run", false, "only say what would be removed")
	downloads := fs.Bool("downloads", false, "also remove the hf:// download cache")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk prune [--dry-run] [--downloads]\n\nDeletes every blob no tagged image reaches, and serving files (exported\nbases and adapters) that no running container uses.")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 {
		return 2
	}
	res, err := client().Prune(context.Background(), api.PruneRequest{DryRun: *dry, Downloads: *downloads})
	if err != nil {
		fmt.Fprintf(stderr, "wpk prune: %v\n", err)
		return 1
	}
	verb := "Removed"
	if *dry {
		verb = "Would remove"
	}
	fmt.Fprintf(stdout, "%s %d blobs, %s; keeping %d blobs, %s\n", verb, res.Blobs, api.FormatSize(res.Bytes), res.KeptBlobs, api.FormatSize(res.KeptBytes))
	if res.ServingBytes > 0 {
		fmt.Fprintf(stdout, "%s %s of serving files no container uses\n", verb, api.FormatSize(res.ServingBytes))
	}
	if *downloads {
		fmt.Fprintf(stdout, "%s %s of hf:// downloads\n", verb, api.FormatSize(res.DownloadBytes))
	}
	return 0
}
