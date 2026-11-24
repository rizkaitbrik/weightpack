package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

// runImport builds a one-layer Weightfile, FROM the parent with the
// fine-tune as an XOR layer, so an import is a build like any other.
func runImport(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("import", flag.ContinueOnError)
	fs.SetOutput(stderr)
	parent := fs.String("parent", "", "image or base model directory the fine-tune was trained from (required)")
	id := fs.String("id", "import", "layer id for the fine-tune")
	var tags stringList
	fs.Var(&tags, "t", "name and optional tag for the image (repeatable)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, `usage: wpk import --parent <image | dir> [--id name] [-t name[:tag]]... <fine-tune dir>

Stores a full fine-tune (a directory of safetensors weights) as a lossless
XOR delta over its parent: only the tensors it changed, compressed. The
same as building:

  FROM <parent>
  LAYER <id> <fine-tune dir> codec=xor`)
		fs.PrintDefaults()
	}
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		return 2
	}
	if len(pos) != 1 || *parent == "" {
		fs.Usage()
		return 2
	}
	src, err := filepath.Abs(pos[0])
	if err != nil {
		fmt.Fprintf(stderr, "wpk import: %v\n", err)
		return 1
	}
	from := *parent
	if info, err := os.Stat(from); err == nil && info.IsDir() {
		if from, err = filepath.Abs(from); err != nil {
			fmt.Fprintf(stderr, "wpk import: %v\n", err)
			return 1
		}
	}

	dir, err := os.MkdirTemp("", "wpk-import-")
	if err != nil {
		fmt.Fprintf(stderr, "wpk import: %v\n", err)
		return 1
	}
	defer os.RemoveAll(dir)
	wf := filepath.Join(dir, "Weightfile")
	body := fmt.Sprintf("FROM %s\nLAYER %s %s codec=xor\n", weightfileValue(from), *id, weightfileValue(src))
	if err := os.WriteFile(wf, []byte(body), 0o600); err != nil {
		fmt.Fprintf(stderr, "wpk import: %v\n", err)
		return 1
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res, err := client().Build(ctx, api.BuildRequest{Weightfile: wf, Tags: tags}, printEvent(stdout, stderr))
	if err != nil {
		fmt.Fprintf(stderr, "wpk import: %s\n", strings.ReplaceAll(err.Error(), wf+":2: ", ""))
		return 1
	}
	fmt.Fprintf(stdout, "Imported %s as %s: %d blobs, %s (%s new)\n",
		pos[0], res.Manifest.Digest, res.Blobs, api.FormatSize(res.Size), api.FormatSize(res.NewBytes))
	for _, t := range res.Tags {
		fmt.Fprintf(stdout, "Tagged %s\n", t)
	}
	return 0
}

// weightfileValue quotes a value for a Weightfile when it needs it.
func weightfileValue(v string) string {
	if v != "" && !strings.ContainsAny(v, " \t\"\\#") {
		return v
	}
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(v) + `"`
}
