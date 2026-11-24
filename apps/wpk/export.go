package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

func runExport(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	fs.SetOutput(stderr)
	out := fs.String("o", "", "directory to write; must not exist yet, or be empty (required)")
	format := fs.String("format", "hf", "output format; only hf is supported so far")
	maxShard := fs.String("max-shard-size", "5GB", "largest base weight file, such as 5GB or 500MB")
	fs.Usage = func() {
		fmt.Fprintln(stderr, `usage: wpk export -o DIR [--format hf] [--max-shard-size 5GB] <ref>

Writes DIR/base (a Hugging Face model directory), DIR/adapters/<id> (a PEFT
adapter per LoRA layer), the image's chat template and license, and
DIR/Weightfile, from which wpk build rebuilds the same image.`)
		fs.PrintDefaults()
	}
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		return 2
	}
	if len(pos) != 1 || *out == "" {
		fs.Usage()
		return 2
	}
	size, err := parseSize(*maxShard)
	if err != nil {
		fmt.Fprintf(stderr, "wpk export: --max-shard-size: %v\n", err)
		return 2
	}
	dir, err := filepath.Abs(*out)
	if err != nil {
		fmt.Fprintf(stderr, "wpk export: %v\n", err)
		return 1
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	req := api.ExportRequest{Ref: pos[0], Dir: dir, Format: *format, MaxShardSize: size}
	res, err := client().Export(ctx, req, printEvent(stdout, stderr))
	if err != nil {
		fmt.Fprintf(stderr, "wpk export: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Exported %s to %s: %d files, %s\n", pos[0], res.Dir, res.Files, api.FormatSize(res.Bytes))
	return 0
}

// parseSize reads a byte count with an optional decimal unit: 5GB, 500MB,
// 64kB or 1000.
func parseSize(s string) (int64, error) {
	units := []struct {
		suffix string
		mult   int64
	}{{"GB", 1e9}, {"MB", 1e6}, {"KB", 1e3}, {"B", 1}}
	upper := strings.ToUpper(strings.TrimSpace(s))
	mult := int64(1)
	for _, u := range units {
		if n, ok := strings.CutSuffix(upper, u.suffix); ok {
			upper, mult = strings.TrimSpace(n), u.mult
			break
		}
	}
	n, err := strconv.ParseFloat(upper, 64)
	if err != nil || n <= 0 {
		return 0, errors.New("want a positive size such as 5GB or 500MB")
	}
	return int64(n * float64(mult)), nil
}
