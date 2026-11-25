// Command wpk is the Weightpack command-line client. It parses commands and
// forwards work to the local wpkd daemon over its Unix socket ($WPK_SOCKET,
// else $WPK_ROOT/wpkd.sock, else ~/.weightpack/wpkd.sock).
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
)

// version is set at build time with -ldflags "-X main.version=...".
var version = "dev"

type command struct {
	summary string
	run     func(args []string, stdout, stderr io.Writer) int
}

// planned commands from the design doc; they exist so the CLI surface is
// visible from day one and fail clearly until implemented.
var planned = map[string]string{
	"diff":   "Show which tensors differ between two images",
	"rebase": "Move layers onto a new base (transfer, gate, distil)",
}

func commands() map[string]command {
	cmds := map[string]command{
		"version":  {"Print the wpk version", runVersion},
		"validate": {"Check a Weightfile or image config against the composition rules", runValidate},
		"build":    {"Build an image from a Weightfile", runBuild},
		"images":   {"List the images in the local store", runImages},
		"inspect":  {"Print an image's manifest and config as JSON", runInspect},
		"history":  {"Show an image's layers, parents, sizes and chain IDs", runHistory},
		"tag":      {"Give a local image another name", runTag},
		"push":     {"Push an image, sending only blobs the registry lacks", runPush},
		"pull":     {"Pull an image, fetching only missing blobs", runPull},
		"sign":     {"Sign an image in a registry with Sigstore (key or keyless)", runSign},
		"verify":   {"Check an image's signatures, as wpk pull does", runVerify},
		"trust":    {"Show or change which signers wpk pull trusts", runTrust},
		"keygen":   {"Make a signing key pair (cosign's format)", runKeygen},
		"squash":   {"Bake an image's LoRA layers into a dense image", runSquash},
		"export":   {"Write an image out as Hugging Face and PEFT files", runExport},
		"import":   {"Store a full fine-tune as a lossless delta over a parent", runImport},
		"rm":       {"Untag images (their blobs stay until prune)", runRm},
		"prune":    {"Delete blobs no tagged image reaches", runPrune},
		"run":      {"Serve an image on the OpenAI-compatible gateway", runRun},
		"ps":       {"List running containers", runPs},
		"stop":     {"Stop containers", runStop},
		"logs":     {"Print a container's engine log", runLogs},
		"doctor":   {"Check that wpkd, the worker, the device and the engines work", runDoctor},
		"service":  {"Install wpkd as a login service (launchd or systemd)", runService},
	}
	for name, summary := range planned {
		cmds[name] = command{summary, notImplemented(name)}
	}
	return cmds
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	cmds := commands()
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		usage(stdout, cmds)
		return 0
	}
	cmd, ok := cmds[args[0]]
	if !ok {
		fmt.Fprintf(stderr, "wpk: unknown command %q\n\n", args[0])
		usage(stderr, cmds)
		return 2
	}
	return cmd.run(args[1:], stdout, stderr)
}

func usage(w io.Writer, cmds map[string]command) {
	fmt.Fprintln(w, "wpk - Docker for neural nets\n\nUsage: wpk <command> [arguments]\n\nCommands:")
	names := make([]string, 0, len(cmds))
	for n := range cmds {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		fmt.Fprintf(w, "  %-9s %s\n", n, cmds[n].summary)
	}
}

func runVersion(_ []string, stdout, _ io.Writer) int {
	fmt.Fprintf(stdout, "wpk %s (config schema %s)\n", version, spec.SchemaV1)
	return 0
}

func runValidate(args []string, stdout, stderr io.Writer) int {
	if len(args) != 1 {
		fmt.Fprintln(stderr, "usage: wpk validate <Weightfile | directory | config.json>")
		return 2
	}
	path := args[0]
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		path = filepath.Join(path, "Weightfile")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "wpk: %v\n", err)
		return 1
	}
	if trimmed := bytes.TrimSpace(data); len(trimmed) > 0 && trimmed[0] == '{' {
		return validateConfig(path, data, stdout, stderr)
	}
	return validateWeightfile(path, data, stdout, stderr)
}

func validateConfig(path string, data []byte, stdout, stderr io.Writer) int {
	var cfg spec.Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		fmt.Fprintf(stderr, "wpk: parse %s: %v\n", path, err)
		return 1
	}
	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(stderr, "wpk: invalid config: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "%s: valid (%d layers)\n", path, len(cfg.Layers))
	return 0
}

func validateWeightfile(path string, data []byte, stdout, stderr io.Writer) int {
	wf, err := weightfile.Parse(bytes.NewReader(data))
	if err == nil {
		_, err = wf.Config()
	}
	if err != nil {
		var we *weightfile.Error
		for _, e := range splitErrors(err) {
			if errors.As(e, &we) {
				we.File = path
			}
			fmt.Fprintf(stderr, "%v\n", e)
		}
		return 1
	}
	fmt.Fprintf(stdout, "%s: valid\n", path)
	fmt.Fprintf(stdout, "  FROM    %s\n", wf.From.Source)
	for _, l := range wf.Layers {
		fmt.Fprintf(stdout, "  LAYER   %-12s %s%s\n", l.ID, l.Source, codecNote(l))
	}
	if m := wf.Mix; m != nil {
		blocks := m.Blocks
		if blocks == "" {
			blocks = "all"
		}
		fmt.Fprintf(stdout, "  MIX     %-12s targets=%s blocks=%s top_k=%d router=%s\n",
			m.ID, strings.Join(m.Targets, ","), blocks, m.TopK, m.Router.Kind)
		for _, e := range m.Experts {
			fmt.Fprintf(stdout, "    EXPERT %-12s %s%s\n", e.ID, e.Source, codecNote(e))
		}
	}
	return 0
}

func codecNote(l weightfile.Layer) string {
	if l.Codec == "" || l.Codec == weightfile.CodecLoRA {
		return ""
	}
	return "  (import, codec=" + l.Codec + ")"
}

func splitErrors(err error) []error {
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		return joined.Unwrap()
	}
	return []error{err}
}

func notImplemented(name string) func([]string, io.Writer, io.Writer) int {
	return func(_ []string, _, stderr io.Writer) int {
		fmt.Fprintf(stderr, "wpk %s: not implemented yet (see docs/design.md)\n", name)
		return 3
	}
}
