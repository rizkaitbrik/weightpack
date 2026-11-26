// Package export writes an image out as plain files that other tools load:
//
//	DIR/Weightfile           `wpk build DIR` rebuilds the same image
//	DIR/base/                a Hugging Face model directory: safetensors
//	                         shards (with model.safetensors.index.json when
//	                         there are several), config.json, tokenizer, …
//	DIR/adapters/<id>/       a PEFT LoRA adapter per LoRA layer:
//	                         adapter_config.json, adapter_model.safetensors
//	DIR/routers/<id>.safetensors  a MIX's trained router, if it has one
//	DIR/chat_template.jinja  the image's TEMPLATE, if it has one
//	DIR/LICENSE              the image's LICENSE file, if it has one
//
// The base keeps only the base layer's files, so rebuilding from the export
// reproduces the base layer, and with it the whole image, byte for byte.
// Export moves bytes and renames tensors; it never reads tensor values.
package export

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
)

// DefaultMaxShardSize is Hugging Face's default max_shard_size, 5 GB.
const DefaultMaxShardSize = 5_000_000_000

// peftPrefix is how PEFT names adapter tensors in a saved adapter.
const peftPrefix = "base_model.model."

// Options say where and how to export.
type Options struct {
	// Dir is the absolute output directory. It must not exist, or be empty.
	Dir string
	// MaxShardSize caps each base weight file; 0 means DefaultMaxShardSize.
	// A tensor larger than the cap gets a shard of its own.
	MaxShardSize int64
	// NoBase and Adapters limit the export to what an engine loads, for
	// serving. With NoBase the base is not written (its tensor table is
	// still read, to describe the adapters); a non-nil Adapters writes only
	// those LoRA layers. A partial export has no Weightfile, since it could
	// not rebuild the image.
	NoBase   bool
	Adapters []string
	// Routers writes a trained MIX's router in a partial export too.
	Routers bool
	// Deltas writes those bit1 layers as the reference engine serves them,
	// over the layers below them (see DeltaDir). It makes the export a
	// partial one.
	Deltas []string
}

// Export writes the image ref names to opts.Dir. It writes into a staging
// directory beside Dir and renames it into place at the end, so a failed
// export leaves nothing behind.
func Export(ctx context.Context, s *store.Store, ref string, opts Options, progress func(api.Event)) (*api.ExportResult, error) {
	if progress == nil {
		progress = func(api.Event) {}
	}
	if opts.MaxShardSize <= 0 {
		opts.MaxShardSize = DefaultMaxShardSize
	}
	if err := checkTarget(opts.Dir); err != nil {
		return nil, err
	}
	root, err := s.Resolve(ref)
	if err != nil {
		return nil, err
	}
	var m ocispec.Manifest
	var cfg spec.Config
	if err := s.ReadJSON(root.Digest.String(), &m); err != nil {
		return nil, err
	}
	if err := s.ReadJSON(m.Config.Digest.String(), &cfg); err != nil {
		return nil, err
	}
	if err := spec.CheckImage(&m, &cfg); err != nil {
		return nil, fmt.Errorf("%s is inconsistent, not exporting it: %w", ref, err)
	}
	plan, err := planLayers(&cfg)
	if err != nil {
		return nil, fmt.Errorf("export %s: %w", ref, err)
	}

	staging, err := os.MkdirTemp(filepath.Dir(opts.Dir), "."+filepath.Base(opts.Dir)+".partial-")
	if err != nil {
		return nil, err
	}
	done := false
	defer func() {
		if !done {
			_ = os.RemoveAll(staging)
		}
	}()

	e := &exporter{ctx: ctx, s: s, dir: staging, progress: progress, step: "export " + ref}
	byLayer := map[string][]ocispec.Descriptor{}
	for _, d := range m.Layers {
		id := d.Annotations[spec.AnnotationLayer]
		byLayer[id] = append(byLayer[id], d)
	}
	partial := opts.NoBase || opts.Adapters != nil || opts.Deltas != nil
	for _, id := range opts.Deltas {
		if !slices.ContainsFunc(plan.stack, func(l spec.Layer) bool { return l.ID == id && l.Kind == spec.KindDeltaBit1 }) {
			return nil, fmt.Errorf("export %s: the image has no bit1 layer %q in its stack", ref, id)
		}
	}
	for _, id := range opts.Adapters {
		if !slices.ContainsFunc(plan.adapters(), func(l spec.Layer) bool { return l.ID == id }) {
			return nil, fmt.Errorf("export %s: the image has no LoRA layer %q", ref, id)
		}
	}
	if err := plan.check(opts.Adapters); err != nil {
		return nil, fmt.Errorf("export %s: %w", ref, err)
	}
	var baseTensors map[string]safetensors.Tensor
	if opts.NoBase {
		baseTensors, err = e.baseTable(byLayer[plan.base.ID])
	} else {
		baseTensors, err = e.base(byLayer[plan.base.ID], opts.MaxShardSize)
	}
	if err != nil {
		return nil, err
	}
	for _, l := range plan.adapters() {
		if opts.Adapters != nil && !slices.Contains(opts.Adapters, l.ID) {
			continue
		}
		if err := e.adapter(l, byLayer[l.ID], baseTensors); err != nil {
			return nil, err
		}
	}
	for _, id := range opts.Deltas {
		if err := e.delta(id, byLayer[id]); err != nil {
			return nil, err
		}
	}
	if m := plan.mix; m != nil && m.Router == spec.RouterTrained && (!partial || opts.Routers) {
		if err := e.copyBlob(RouterPath(m.ID), byLayer[m.ID][0]); err != nil {
			return nil, err
		}
	}
	files, err := e.imageMeta(byLayer[""])
	if err != nil {
		return nil, err
	}
	if !partial {
		if err := e.weightfile(ref, root.Digest.String(), plan, files, m.Annotations); err != nil {
			return nil, err
		}
	}

	// An empty target directory is replaced; checkTarget made sure it is
	// empty, and rename needs it gone.
	if err := os.Remove(opts.Dir); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if err := os.Rename(staging, opts.Dir); err != nil {
		return nil, err
	}
	done = true
	progress(api.Event{Step: e.step, Message: fmt.Sprintf("wrote %d files, %s, to %s", e.files, api.FormatSize(e.bytes), opts.Dir)})
	return &api.ExportResult{Dir: opts.Dir, Files: e.files, Bytes: e.bytes}, nil
}

// checkTarget accepts an absolute path that does not exist or is an empty
// directory.
func checkTarget(dir string) error {
	if !filepath.IsAbs(dir) {
		return fmt.Errorf("export directory must be an absolute path, got %q", dir)
	}
	entries, err := os.ReadDir(dir)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if info, err := os.Stat(filepath.Dir(dir)); err != nil || !info.IsDir() {
			return fmt.Errorf("export directory's parent %s does not exist", filepath.Dir(dir))
		}
		return nil
	case err != nil:
		return fmt.Errorf("export directory %s: %w", dir, err)
	case len(entries) > 0:
		return fmt.Errorf("export directory %s is not empty", dir)
	}
	return nil
}

// layerPlan is an image's layers in Weightfile form: one stack, and at most
// one mix on top of it.
type layerPlan struct {
	base    spec.Layer
	stack   []spec.Layer
	mix     *spec.Layer
	experts []spec.Layer
}

func (p *layerPlan) adapters() []spec.Layer { return append(slices.Clone(p.stack), p.experts...) }

func planLayers(c *spec.Config) (*layerPlan, error) {
	p := &layerPlan{}
	byID := map[string]spec.Layer{}
	children := map[string][]spec.Layer{}
	isExpert := map[string]bool{}
	for _, l := range c.Layers {
		byID[l.ID] = l
		if l.Kind == spec.KindBase {
			p.base = l
		} else {
			children[l.Parent] = append(children[l.Parent], l)
		}
		if l.Kind == spec.KindMix {
			for _, e := range l.Experts {
				isExpert[e] = true
			}
		}
	}
	for _, l := range c.Layers {
		if !weightfile.ValidID(l.ID) {
			return nil, fmt.Errorf("layer id %q cannot name an export directory", l.ID)
		}
	}
	top := p.base
	for {
		var next []spec.Layer
		for _, ch := range children[top.ID] {
			if !isExpert[ch.ID] {
				next = append(next, ch)
			}
		}
		if len(next) == 0 {
			break
		}
		if len(next) > 1 {
			return nil, fmt.Errorf("layers %s all sit on %s, which a Weightfile cannot express", ids(next), top.ID)
		}
		l := next[0]
		if l.Kind == spec.KindMix {
			p.mix = &l
			if len(children[l.ID]) > 0 {
				return nil, fmt.Errorf("layers sit on MIX %s, which a Weightfile cannot express", l.ID)
			}
			for _, e := range l.Experts {
				p.experts = append(p.experts, byID[e])
			}
			break
		}
		p.stack = append(p.stack, l)
		top = l
	}
	return p, nil
}

// check refuses layers the export would write that have no PEFT form:
// every adapter for a full export, only the named ones for a partial one,
// whose caller (serving) has baked the rest into a base already.
func (p *layerPlan) check(only []string) error {
	for _, l := range p.adapters() {
		if only != nil && !slices.Contains(only, l.ID) {
			continue
		}
		if l.Kind != spec.KindLoRA {
			return fmt.Errorf("layer %q is %s, which has no PEFT form; wpk squash the image and export the result to get dense weights", l.ID, l.Kind)
		}
	}
	return nil
}

func ids(ls []spec.Layer) string {
	var out []string
	for _, l := range ls {
		out = append(out, l.ID)
	}
	return strings.Join(out, ", ")
}

type exporter struct {
	ctx      context.Context
	s        *store.Store
	dir      string
	progress func(api.Event)
	step     string
	files    int
	bytes    int64
}

// writeFile copies r to dir/name.
func (e *exporter) writeFile(name string, r io.Reader) error {
	if err := e.ctx.Err(); err != nil {
		return err
	}
	p := filepath.Join(e.dir, name)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(p, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	n, err := io.Copy(f, r)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("write %s: %w", name, err)
	}
	e.files++
	e.bytes += n
	return nil
}

func (e *exporter) writeJSON(name string, v any) error {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return e.writeFile(name, strings.NewReader(string(raw)+"\n"))
}

func (e *exporter) copyBlob(name string, d ocispec.Descriptor) error {
	rc, err := e.s.Open(d.Digest.String())
	if err != nil {
		return err
	}
	defer rc.Close()
	return e.writeFile(name, rc)
}

type located struct {
	file *safetensors.File
	t    safetensors.Tensor
}

// openTensors opens the tensor blobs among ds and lists their tensors by
// name. The caller closes the returned files.
func (e *exporter) openTensors(ds []ocispec.Descriptor, mediaType string) (map[string]located, []*safetensors.File, error) {
	out := map[string]located{}
	var files []*safetensors.File
	for _, d := range ds {
		if spec.BaseMediaType(d.MediaType) != mediaType {
			continue
		}
		p, err := e.s.Path(d.Digest.String())
		if err != nil {
			return nil, files, err
		}
		f, err := safetensors.Open(p)
		if err != nil {
			return nil, files, err
		}
		files = append(files, f)
		for _, t := range f.Tensors {
			out[t.Name] = located{f, t}
		}
	}
	return out, files, nil
}

func closeAll(files []*safetensors.File) {
	for _, f := range files {
		_ = f.Close()
	}
}

// baseTable reads the base layer's tensor table without writing anything.
func (e *exporter) baseTable(ds []ocispec.Descriptor) (map[string]safetensors.Tensor, error) {
	tensors, files, err := e.openTensors(ds, spec.MediaTypeTensor)
	defer closeAll(files)
	if err != nil {
		return nil, err
	}
	out := make(map[string]safetensors.Tensor, len(tensors))
	for n, l := range tensors {
		out[n] = l.t
	}
	return out, nil
}

// base writes the base layer as a Hugging Face model directory and returns
// its tensor table, which adapters are described against.
func (e *exporter) base(ds []ocispec.Descriptor, maxShard int64) (map[string]safetensors.Tensor, error) {
	tensors, files, err := e.openTensors(ds, spec.MediaTypeTensor)
	defer closeAll(files)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)

	// Fill shards in name order, starting a new one when the next tensor
	// would push the current one past the cap.
	var shards [][]string
	var size, total int64
	for _, n := range names {
		ts := tensors[n].t.Size
		if len(shards) == 0 || (size > 0 && size+ts > maxShard) {
			shards = append(shards, nil)
			size = 0
		}
		shards[len(shards)-1] = append(shards[len(shards)-1], n)
		size += ts
		total += ts
	}
	weightMap := map[string]string{}
	for i, shard := range shards {
		file := "model.safetensors"
		if len(shards) > 1 {
			file = fmt.Sprintf("model-%05d-of-%05d.safetensors", i+1, len(shards))
		}
		var entries []safetensors.Entry
		for _, n := range shard {
			l := tensors[n]
			entries = append(entries, safetensors.Entry{Name: n, DType: l.t.DType, Shape: l.t.Shape, Data: l.file.Data(l.t)})
			weightMap[n] = file
		}
		r, _, err := safetensors.NewReader(entries, map[string]string{"format": "pt"})
		if err != nil {
			return nil, err
		}
		if err := e.writeFile(filepath.Join("base", file), r); err != nil {
			return nil, err
		}
	}
	if len(shards) > 1 {
		index := map[string]any{"metadata": map[string]any{"total_size": total}, "weight_map": weightMap}
		if err := e.writeJSON("base/model.safetensors.index.json", index); err != nil {
			return nil, err
		}
	}
	for _, d := range ds {
		if spec.BaseMediaType(d.MediaType) != spec.MediaTypeMeta {
			continue
		}
		title := d.Annotations[spec.AnnotationTitle]
		if err := checkTitle(title); err != nil {
			return nil, err
		}
		if strings.HasPrefix(title, "model") && strings.Contains(title, ".safetensors") {
			return nil, fmt.Errorf("base meta file %q collides with the weight files", title)
		}
		if err := e.copyBlob(filepath.Join("base", title), d); err != nil {
			return nil, err
		}
	}
	e.progress(api.Event{Step: e.step, Message: fmt.Sprintf("base: %d tensors in %d shards, %s", len(names), len(shards), api.FormatSize(total))})

	out := make(map[string]safetensors.Tensor, len(tensors))
	for n, l := range tensors {
		out[n] = l.t
	}
	return out, nil
}

// checkTitle accepts a plain file name, so a meta blob cannot write outside
// the export directory.
func checkTitle(title string) error {
	if title == "" || title != filepath.Base(title) || title == "." || title == ".." || strings.ContainsAny(title, `/\`) {
		return fmt.Errorf("meta file name %q is not a plain file name", title)
	}
	return nil
}

// peftConfig is the adapter_config.json written for each LoRA layer: what
// PEFT and vLLM read to load it.
type peftConfig struct {
	PeftType string `json:"peft_type"`
	// TaskType stays null: the image does not record the adapter's task,
	// and null makes PEFT load it with its generic wrapper, which suits
	// every model (CAUSAL_LM would break encoders, for one).
	TaskType          *string        `json:"task_type"`
	BaseModel         *string        `json:"base_model_name_or_path"`
	R                 int            `json:"r"`
	LoraAlpha         float64        `json:"lora_alpha"`
	UseRSLoRA         bool           `json:"use_rslora"`
	TargetModules     any            `json:"target_modules"`
	LoraDropout       float64        `json:"lora_dropout"`
	Bias              string         `json:"bias"`
	FanInFanOut       bool           `json:"fan_in_fan_out"`
	UseDoRA           bool           `json:"use_dora"`
	ModulesToSave     []string       `json:"modules_to_save"`
	RankPattern       map[string]any `json:"rank_pattern"`
	AlphaPattern      map[string]any `json:"alpha_pattern"`
	InferenceMode     bool           `json:"inference_mode"`
	InitLoraWeights   bool           `json:"init_lora_weights"`
	LayersToTransform []int          `json:"layers_to_transform"`
}

func (e *exporter) adapter(l spec.Layer, ds []ocispec.Descriptor, base map[string]safetensors.Tensor) error {
	tensors, files, err := e.openTensors(ds, spec.MediaTypeLoRA)
	defer closeAll(files)
	if err != nil {
		return err
	}
	// modules_to_save copies: full tensors stored under base tensor names.
	saved, savedFiles, err := e.openTensors(ds, spec.MediaTypeTensor)
	defer closeAll(savedFiles)
	if err != nil {
		return err
	}
	var savedModules []string
	for n, lt := range saved {
		tensors[n] = lt
		module := n[:max(strings.LastIndex(n, "."), 0)]
		if !slices.Contains(savedModules, module) {
			savedModules = append(savedModules, module)
		}
	}
	sort.Strings(savedModules)
	names := make([]string, 0, len(tensors))
	modules := map[string]bool{}
	for n := range tensors {
		names = append(names, n)
		if m, ok := strings.CutSuffix(n, ".lora_A.weight"); ok {
			modules[m] = true
		}
	}
	sort.Strings(names)
	var entries []safetensors.Entry
	for _, n := range names {
		lt := tensors[n]
		entries = append(entries, safetensors.Entry{Name: peftPrefix + n, DType: lt.t.DType, Shape: lt.t.Shape, Data: lt.file.Data(lt.t)})
	}
	r, _, err := safetensors.NewReader(entries, map[string]string{"format": "pt"})
	if err != nil {
		return err
	}
	dir := filepath.Join("adapters", l.ID)
	if err := e.writeFile(filepath.Join(dir, "adapter_model.safetensors"), r); err != nil {
		return err
	}
	cfg := peftConfig{
		PeftType: "LORA", R: l.Rank, LoraAlpha: l.Alpha, UseRSLoRA: l.RSLoRA,
		TargetModules: targetModules(modules, base), Bias: "none",
		RankPattern: map[string]any{}, AlphaPattern: map[string]any{},
		InferenceMode: true, InitLoraWeights: true, ModulesToSave: savedModules,
	}
	if err := e.writeJSON(filepath.Join(dir, "adapter_config.json"), cfg); err != nil {
		return err
	}
	e.progress(api.Event{Step: e.step, Message: fmt.Sprintf("adapter %s: LoRA rank %d, %d modules", l.ID, l.Rank, len(modules))})
	return nil
}

// targetModules describes an adapter's modules the way PEFT matches them.
// When the adapter covers every base matrix whose module name ends in one
// of its suffixes (q_proj, v_proj, ...), the plain suffix list says exactly
// that. Otherwise a regular expression that PEFT full-matches against module
// names lists them one by one, so no extra module gets a LoRA.
func targetModules(modules map[string]bool, base map[string]safetensors.Tensor) any {
	suffixes := map[string]bool{}
	for m := range modules {
		suffixes[m[strings.LastIndex(m, ".")+1:]] = true
	}
	exact := true
	for name, t := range base {
		m, ok := strings.CutSuffix(name, ".weight")
		if !ok || len(t.Shape) != 2 || !suffixes[m[strings.LastIndex(m, ".")+1:]] {
			continue
		}
		if !modules[m] {
			exact = false
			break
		}
	}
	if exact {
		list := make([]string, 0, len(suffixes))
		for s := range suffixes {
			list = append(list, s)
		}
		sort.Strings(list)
		return list
	}
	list := make([]string, 0, len(modules))
	for m := range modules {
		list = append(list, regexp.QuoteMeta(m))
	}
	sort.Strings(list)
	return "^(?:" + strings.Join(list, "|") + ")$"
}

// imageMeta writes image-level meta files (TEMPLATE, a LICENSE file) to the
// export root and returns their titles.
func (e *exporter) imageMeta(ds []ocispec.Descriptor) (map[string]bool, error) {
	files := map[string]bool{}
	for _, d := range ds {
		title := d.Annotations[spec.AnnotationTitle]
		if err := checkTitle(title); err != nil {
			return nil, err
		}
		if title == "Weightfile" || title == "base" || title == "adapters" || title == "routers" {
			return nil, fmt.Errorf("image meta file %q collides with the export layout", title)
		}
		if err := e.copyBlob(title, d); err != nil {
			return nil, err
		}
		files[title] = true
	}
	return files, nil
}

// weightfile writes the Weightfile that rebuilds the image from the export.
// The manifest's annotations come back as LICENSE (the SPDX licenses one)
// and LABEL lines (all others, such as a squash's provenance), so the
// rebuilt manifest is the same.
func (e *exporter) weightfile(ref, digest string, p *layerPlan, files map[string]bool, annotations map[string]string) error {
	var b strings.Builder
	fmt.Fprintf(&b, "# Exported from %s (%s) by wpk export.\n# `wpk build` on this directory rebuilds the same layers.\n", ref, digest)
	b.WriteString("FROM ./base\n")
	for _, l := range p.stack {
		fmt.Fprintf(&b, "LAYER %s ./adapters/%s\n", l.ID, l.ID)
	}
	if m := p.mix; m != nil {
		fmt.Fprintf(&b, "MIX %s targets=%s", m.ID, strings.Join(m.Targets, ","))
		if m.Blocks != "" {
			fmt.Fprintf(&b, " blocks=%s", m.Blocks)
		}
		fmt.Fprintf(&b, " top_k=%d\n", m.TopK)
		for _, x := range p.experts {
			fmt.Fprintf(&b, "  EXPERT %s ./adapters/%s\n", x.ID, x.ID)
		}
		switch m.Router {
		case spec.RouterArrow:
			b.WriteString("  ROUTER arrow\n")
		case spec.RouterTrained:
			fmt.Fprintf(&b, "  ROUTER ./%s\n", RouterPath(m.ID))
		default:
			return fmt.Errorf("MIX %s uses router %q, which export cannot write", m.ID, m.Router)
		}
	}
	if files["chat_template.jinja"] {
		b.WriteString("TEMPLATE ./chat_template.jinja\n")
	}
	license := annotations[ocispec.AnnotationLicenses]
	switch {
	case files["LICENSE"]:
		b.WriteString("LICENSE ./LICENSE\n")
	case license != "":
		v, err := quote(license)
		if err != nil {
			return fmt.Errorf("license %w", err)
		}
		fmt.Fprintf(&b, "LICENSE %s\n", v)
	}
	keys := make([]string, 0, len(annotations))
	for k := range annotations {
		if k != ocispec.AnnotationLicenses {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	for _, k := range keys {
		if !labelKey.MatchString(k) {
			return fmt.Errorf("annotation key %q cannot be written as a Weightfile LABEL", k)
		}
		v, err := quote(annotations[k])
		if err != nil {
			return fmt.Errorf("annotation %s %w", k, err)
		}
		if v == "" {
			v = `""`
		}
		fmt.Fprintf(&b, "LABEL %s=%s\n", k, v)
	}
	// The Weightfile must parse; anything else is a bug here.
	if _, err := weightfile.Parse(strings.NewReader(b.String())); err != nil {
		return fmt.Errorf("export wrote an invalid Weightfile (this is a bug): %w", err)
	}
	return e.writeFile("Weightfile", strings.NewReader(b.String()))
}

// labelKey matches the keys a Weightfile LABEL accepts.
var labelKey = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$`)

// quote writes a Weightfile value: as it is when it is a plain word, else
// in double quotes with the only escapes the lexer knows, \" and \\. A
// value with a line break cannot be written at all.
func quote(v string) (string, error) {
	if strings.ContainsAny(v, "\r\n") {
		return "", errors.New("value has a line break, which a Weightfile cannot carry")
	}
	if v != "" && !strings.ContainsAny(v, " \t\"\\#") {
		return v, nil
	}
	if v == "" {
		return "", nil
	}
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(v) + `"`, nil
}

// RouterPath is where an export puts a MIX's trained router, relative to
// the export directory.
func RouterPath(mix string) string { return filepath.Join("routers", mix+".safetensors") }

// DeltaDir is where an export puts a bit1 layer for the reference engine,
// relative to the export directory.
func DeltaDir(layer string) string { return filepath.Join("deltas", layer) }

// DeltaFile describes a bit1 layer's directory for the reference engine,
// which loads it as it loads an adapter (python/worker deltas.py).
const DeltaFile = "weightpack_delta.json"

// deltaMeta is DeltaFile's content, version 1.
type deltaMeta struct {
	Version int      `json:"version"`
	Codec   string   `json:"codec"`
	Layer   string   `json:"layer"`
	Files   []string `json:"files"`
}

// delta writes a bit1 layer's blobs as they are stored, one tensor's delta
// each (the signs and scale of a matrix, a whole float32 delta of any other
// tensor), in tensor name order, and DeltaFile naming them. The engine
// reads them as decoding does; the daemon does no tensor work. A layer
// whose fine-tune changed nothing has no blobs, and its file names none.
func (e *exporter) delta(id string, ds []ocispec.Descriptor) error {
	ds = slices.Clone(ds)
	sort.Slice(ds, func(i, j int) bool {
		return ds[i].Annotations[spec.AnnotationTensor] < ds[j].Annotations[spec.AnnotationTensor]
	})
	meta := deltaMeta{Version: 1, Codec: "bit1", Layer: id, Files: []string{}}
	for _, d := range ds {
		if d.MediaType != spec.MediaTypeDeltaBit1 {
			return fmt.Errorf("layer %q holds a %s blob, not a bit1 delta", id, d.MediaType)
		}
		name := fmt.Sprintf("%06d.safetensors", len(meta.Files))
		if err := e.copyBlob(filepath.Join(DeltaDir(id), name), d); err != nil {
			return err
		}
		meta.Files = append(meta.Files, name)
	}
	// vLLM reads an adapter config before its loader, which the
	// weightpack-vllm plugin replaces for this directory, runs: a rank-1
	// adapter with no LoRA weights.
	adapter := map[string]any{
		"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 1, "lora_alpha": 1,
		"target_modules": []string{}, "bias": "none", "use_rslora": false,
		"fan_in_fan_out": false, "base_model_name_or_path": "",
	}
	if err := e.writeJSON(filepath.Join(DeltaDir(id), "adapter_config.json"), adapter); err != nil {
		return err
	}
	return e.writeJSON(filepath.Join(DeltaDir(id), DeltaFile), meta)
}
