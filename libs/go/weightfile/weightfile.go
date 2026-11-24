// Package weightfile parses Weightfiles, the build files of Weightpack
// images, and turns them into an image config.
//
// A Weightfile reads top to bottom. FROM names the base; each LAYER is
// stacked on everything above it; a MIX puts the EXPERT lines that follow it
// side by side behind a ROUTER. See docs/weightfile.md for the full grammar.
package weightfile

import (
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// Error is a problem at a line of a Weightfile.
type Error struct {
	File string
	Line int
	Msg  string
}

func (e *Error) Error() string {
	name := e.File
	if name == "" {
		name = "Weightfile"
	}
	return fmt.Sprintf("%s:%d: %s", name, e.Line, e.Msg)
}

// File is a parsed Weightfile.
type File struct {
	From     From
	Layers   []Layer // stacked layers, bottom to top
	Mix      *Mix    // optional; always on top of the stack
	Template *Meta   // chat template path
	License  *Meta   // "inherit", an SPDX id or a path
}

// From is the FROM instruction.
type From struct {
	Line   int
	Source Source
}

// Layer is a LAYER, or an EXPERT inside a MIX.
type Layer struct {
	Line   int
	ID     string
	Source Source
	// Codec is "" or "lora" for an adapter. "xor", "bit1" or "sparse" means
	// the source is a full fine-tune to import as a delta.
	Codec   string
	Eval    string   // eval suite for the fidelity gate; "" = domain default
	MaxDrop *float64 // maximum eval drop a lossy import may cost
}

// Mix is a MIX block.
type Mix struct {
	Line    int
	ID      string
	Targets []string // canonical module names, e.g. q, k, v, o, gate, up, down
	Blocks  string   // "" for every block, else ranges such as "8-31"
	TopK    int
	Experts []Layer
	Router  Router
}

// Router is the ROUTER line of a MIX. Kind is "arrow" (zero-shot) or
// "train" (experts frozen, router trained on Data for Steps steps).
type Router struct {
	Line  int
	Kind  string
	Data  string
	Steps int
}

// Meta is a TEMPLATE or LICENSE value.
type Meta struct {
	Line  int
	Value string
}

// Codecs accepted by the codec= parameter.
const (
	CodecLoRA   = "lora"
	CodecXOR    = "xor"
	CodecBit1   = "bit1"
	CodecSparse = "sparse"
)

// Module names a MIX may target, in canonical order, and the aliases that
// expand to them.
var (
	modules       = []string{"q", "k", "v", "o", "gate", "up", "down"}
	moduleAliases = map[string][]string{
		"attn": {"q", "k", "v", "o"},
		"mlp":  {"gate", "up", "down"},
		"all":  modules,
	}
)

// ReservedBaseID is the layer id the FROM line gets in the image config.
const ReservedBaseID = "base"

var idPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)

// ValidID reports whether id is a valid layer id: lowercase letters, digits,
// '.', '_' and '-', starting with a letter or digit, at most 64 characters.
func ValidID(id string) bool { return idPattern.MatchString(id) }

// ParseFile parses the Weightfile at path.
func ParseFile(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	wf, err := Parse(f)
	if err != nil {
		return nil, withFile(err, path)
	}
	return wf, nil
}

func withFile(err error, path string) error {
	var errs []error
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		errs = joined.Unwrap()
	} else {
		errs = []error{err}
	}
	for _, e := range errs {
		var we *Error
		if errors.As(e, &we) {
			we.File = path
		}
	}
	return err
}

// Parse reads a Weightfile. All errors found are returned together, joined,
// each an *Error carrying its line number.
func Parse(r io.Reader) (*File, error) {
	lines, errs := lex(r)
	p := &parser{}
	for _, l := range lines {
		p.line(l)
	}
	p.finish()
	errs = append(errs, p.errs...)
	if len(errs) > 0 {
		slices.SortStableFunc(errs, func(a, b error) int {
			var ea, eb *Error
			errors.As(a, &ea)
			errors.As(b, &eb)
			if ea == nil || eb == nil {
				return 0
			}
			return ea.Line - eb.Line
		})
		return nil, errors.Join(errs...)
	}
	return &p.file, nil
}

type parser struct {
	file    File
	sawFrom bool
	inMix   bool // a MIX block is open: EXPERT and ROUTER attach to it
	ids     map[string]int
	errs    []error
}

func (p *parser) errorf(line int, format string, args ...any) {
	p.errs = append(p.errs, &Error{Line: line, Msg: fmt.Sprintf(format, args...)})
}

func (p *parser) line(l rawLine) {
	instr := strings.ToUpper(l.fields[0])
	args := l.fields[1:]

	if !p.sawFrom && instr != "FROM" {
		p.errorf(l.no, "%s before FROM: a Weightfile starts with FROM", instr)
		p.sawFrom = true // report once
	}
	if instr != "EXPERT" && instr != "ROUTER" {
		p.inMix = false
	}

	switch instr {
	case "FROM":
		p.from(l, args)
	case "LAYER":
		p.layer(l, args)
	case "MIX":
		p.mix(l, args)
	case "EXPERT":
		p.expert(l, args)
	case "ROUTER":
		p.router(l, args)
	case "TEMPLATE":
		p.meta(l, args, &p.file.Template, "TEMPLATE", "<path>")
	case "LICENSE":
		p.meta(l, args, &p.file.License, "LICENSE", "inherit | <SPDX id> | <path>")
	case "TRAIN":
		p.errorf(l.no, "TRAIN blocks are not supported yet; train the adapter and give its path")
	default:
		p.errorf(l.no, "unknown instruction %q", l.fields[0])
	}
}

func (p *parser) from(l rawLine, args []string) {
	if p.sawFrom && p.file.From.Line != 0 {
		p.errorf(l.no, "second FROM (first at line %d): an image has one base", p.file.From.Line)
		return
	}
	p.sawFrom = true
	if len(args) != 1 {
		p.errorf(l.no, "usage: FROM <reference>")
		return
	}
	src, err := ParseSource(args[0])
	if err != nil {
		p.errorf(l.no, "FROM: %v", err)
		return
	}
	p.file.From = From{Line: l.no, Source: src}
}

func (p *parser) claimID(line int, id string) bool {
	if p.ids == nil {
		p.ids = map[string]int{}
	}
	switch {
	case id == ReservedBaseID:
		p.errorf(line, "layer id %q is reserved for the FROM layer", id)
		return false
	case !idPattern.MatchString(id):
		p.errorf(line, "bad layer id %q: use lowercase letters, digits, '.', '_' or '-' (max 64)", id)
		return false
	}
	if prev, dup := p.ids[id]; dup {
		p.errorf(line, "layer id %q already used at line %d", id, prev)
		return false
	}
	p.ids[id] = line
	return true
}

// weightLayer parses "<id> <source> [codec=] [eval=] [max_drop=]".
func (p *parser) weightLayer(l rawLine, args []string, instr string) (Layer, bool) {
	if len(args) < 2 {
		p.errorf(l.no, "usage: %s <id> <source> [codec=lora|xor|bit1|sparse] [eval=<suite>] [max_drop=<n>]", instr)
		return Layer{}, false
	}
	ok := p.claimID(l.no, args[0])
	layer := Layer{Line: l.no, ID: args[0]}
	if args[1] == "TRAIN" {
		p.errorf(l.no, "TRAIN blocks are not supported yet; train the adapter and give its path")
		return Layer{}, false
	}
	src, err := ParseSource(args[1])
	if err != nil {
		p.errorf(l.no, "%s %s: %v", instr, args[0], err)
		ok = false
	}
	layer.Source = src

	params, err := parseParams(args[2:], "codec", "eval", "max_drop")
	if err != nil {
		p.errorf(l.no, "%s %s: %v", instr, args[0], err)
		return layer, false
	}
	layer.Codec = params["codec"]
	layer.Eval = params["eval"]
	switch layer.Codec {
	case "", CodecLoRA, CodecXOR, CodecBit1, CodecSparse:
	default:
		p.errorf(l.no, "%s %s: unknown codec %q (want lora, xor, bit1 or sparse)", instr, args[0], layer.Codec)
		ok = false
	}
	if v, set := params["max_drop"]; set {
		d, err := strconv.ParseFloat(v, 64)
		if err != nil || d < 0 {
			p.errorf(l.no, "%s %s: max_drop must be a number >= 0, got %q", instr, args[0], v)
			ok = false
		} else {
			layer.MaxDrop = &d
		}
	}
	lossy := layer.Codec == CodecBit1 || layer.Codec == CodecSparse
	switch {
	case lossy && layer.MaxDrop == nil:
		p.errorf(l.no, "%s %s: codec=%s is lossy and needs max_drop=<n>", instr, args[0], layer.Codec)
		ok = false
	case !lossy && (layer.MaxDrop != nil || layer.Eval != ""):
		p.errorf(l.no, "%s %s: eval and max_drop apply only to lossy codecs (bit1, sparse)", instr, args[0])
		ok = false
	}
	return layer, ok
}

func (p *parser) layer(l rawLine, args []string) {
	if p.file.Mix != nil {
		p.errorf(l.no, "LAYER after MIX (line %d): stack layers below the mix", p.file.Mix.Line)
		return
	}
	if layer, ok := p.weightLayer(l, args, "LAYER"); ok {
		p.file.Layers = append(p.file.Layers, layer)
	}
}

func (p *parser) mix(l rawLine, args []string) {
	if p.file.Mix != nil {
		p.errorf(l.no, "second MIX (first at line %d): an image has at most one mix for now", p.file.Mix.Line)
		return
	}
	if len(args) < 1 {
		p.errorf(l.no, "usage: MIX <id> [targets=mlp|attn|all|q,k,...] [blocks=all|8-31] [top_k=<n>]")
		return
	}
	m := &Mix{Line: l.no, ID: args[0]}
	p.file.Mix = m
	p.inMix = true
	p.claimID(l.no, args[0])

	params, err := parseParams(args[1:], "targets", "blocks", "top_k")
	if err != nil {
		p.errorf(l.no, "MIX %s: %v", m.ID, err)
		return
	}
	if m.Targets, err = parseTargets(params["targets"]); err != nil {
		p.errorf(l.no, "MIX %s: %v", m.ID, err)
	}
	if m.Blocks, err = parseBlocks(params["blocks"]); err != nil {
		p.errorf(l.no, "MIX %s: %v", m.ID, err)
	}
	if v, set := params["top_k"]; set {
		k, err := strconv.Atoi(v)
		if err != nil || k < 1 {
			p.errorf(l.no, "MIX %s: top_k must be an integer >= 1, got %q", m.ID, v)
		} else {
			m.TopK = k
		}
	}
}

func (p *parser) expert(l rawLine, args []string) {
	if !p.inMix {
		p.errorf(l.no, "EXPERT outside a MIX block: put it right after the MIX line")
		return
	}
	if layer, ok := p.weightLayer(l, args, "EXPERT"); ok {
		p.file.Mix.Experts = append(p.file.Mix.Experts, layer)
	}
}

func (p *parser) router(l rawLine, args []string) {
	if !p.inMix {
		p.errorf(l.no, "ROUTER outside a MIX block: put it after the mix's EXPERT lines")
		return
	}
	m := p.file.Mix
	if m.Router.Line != 0 {
		p.errorf(l.no, "second ROUTER for MIX %s (first at line %d)", m.ID, m.Router.Line)
		return
	}
	if len(args) < 1 {
		p.errorf(l.no, "usage: ROUTER arrow | ROUTER train data=<path> steps=<n>")
		return
	}
	r := Router{Line: l.no, Kind: strings.ToLower(args[0])}
	switch r.Kind {
	case "arrow":
		if len(args) > 1 {
			p.errorf(l.no, "ROUTER arrow takes no parameters")
			return
		}
	case "train":
		params, err := parseParams(args[1:], "data", "steps")
		if err != nil {
			p.errorf(l.no, "ROUTER train: %v", err)
			return
		}
		r.Data = params["data"]
		if r.Data == "" {
			p.errorf(l.no, "ROUTER train needs data=<path>")
			return
		}
		r.Steps = 2000
		if v, set := params["steps"]; set {
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				p.errorf(l.no, "ROUTER train: steps must be an integer >= 1, got %q", v)
				return
			}
			r.Steps = n
		}
	default:
		p.errorf(l.no, "unknown router %q (want arrow or train)", args[0])
		return
	}
	m.Router = r
}

func (p *parser) meta(l rawLine, args []string, dst **Meta, instr, usage string) {
	if *dst != nil {
		p.errorf(l.no, "second %s (first at line %d)", instr, (*dst).Line)
		return
	}
	if len(args) != 1 {
		p.errorf(l.no, "usage: %s %s", instr, usage)
		return
	}
	*dst = &Meta{Line: l.no, Value: args[0]}
}

// finish applies defaults and the checks that need the whole file.
func (p *parser) finish() {
	if p.file.From.Line == 0 {
		if len(p.errs) == 0 {
			p.errs = append(p.errs, &Error{Line: 1, Msg: "missing FROM"})
		}
		return
	}
	m := p.file.Mix
	if m == nil {
		return
	}
	if len(m.Experts) == 0 {
		p.errorf(m.Line, "MIX %s has no EXPERT lines", m.ID)
		return
	}
	if m.Targets == nil {
		m.Targets = slices.Clone(modules)
	}
	if m.TopK == 0 {
		m.TopK = min(2, len(m.Experts))
	}
	if m.TopK > len(m.Experts) {
		p.errorf(m.Line, "MIX %s: top_k=%d but only %d experts", m.ID, m.TopK, len(m.Experts))
	}
	if m.Router.Kind == "" {
		m.Router = Router{Line: m.Line, Kind: "arrow"}
	}
}

func parseParams(fields []string, allowed ...string) (map[string]string, error) {
	out := make(map[string]string, len(fields))
	for _, f := range fields {
		k, v, ok := strings.Cut(f, "=")
		if !ok || k == "" {
			return nil, fmt.Errorf("unexpected %q: want key=value (one of %s)", f, strings.Join(allowed, ", "))
		}
		k = strings.ToLower(k)
		if !slices.Contains(allowed, k) {
			return nil, fmt.Errorf("unknown parameter %q (want %s)", k, strings.Join(allowed, ", "))
		}
		if _, dup := out[k]; dup {
			return nil, fmt.Errorf("parameter %q given twice", k)
		}
		if v == "" {
			return nil, fmt.Errorf("parameter %q has no value", k)
		}
		out[k] = v
	}
	return out, nil
}

// parseTargets expands aliases and returns module names in canonical order.
// An empty value returns nil, meaning "use the default".
func parseTargets(v string) ([]string, error) {
	if v == "" {
		return nil, nil
	}
	set := map[string]bool{}
	for _, t := range strings.Split(v, ",") {
		t = strings.ToLower(strings.TrimSpace(t))
		if expanded, ok := moduleAliases[t]; ok {
			for _, m := range expanded {
				set[m] = true
			}
			continue
		}
		if !slices.Contains(modules, t) {
			return nil, fmt.Errorf("unknown target %q (want mlp, attn, all or %s)", t, strings.Join(modules, ", "))
		}
		set[t] = true
	}
	var out []string
	for _, m := range modules {
		if set[m] {
			out = append(out, m)
		}
	}
	return out, nil
}

// parseBlocks checks a block selection: "all", or ascending,
// non-overlapping ranges such as "0-3,8-31". "all" and "" return "".
func parseBlocks(v string) (string, error) {
	if v == "" || strings.EqualFold(v, "all") {
		return "", nil
	}
	prev := -1
	parts := strings.Split(v, ",")
	for i, part := range parts {
		lo, hi, isRange := strings.Cut(part, "-")
		a, errA := strconv.Atoi(lo)
		b := a
		var errB error
		if isRange {
			b, errB = strconv.Atoi(hi)
		}
		if errA != nil || errB != nil || a < 0 || b < a {
			return "", fmt.Errorf("bad blocks range %q (want all, or ranges like 8-31)", part)
		}
		if a <= prev {
			return "", fmt.Errorf("blocks %q: ranges must ascend without overlap", v)
		}
		prev = b
		parts[i] = strings.TrimSpace(part)
	}
	return strings.Join(parts, ","), nil
}
