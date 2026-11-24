// Package safetensors reads safetensors headers and writes safetensors files
// in a canonical byte layout.
//
// It never decodes tensor values. Weightpack's Go side moves tensors as byte
// ranges named by the header; any math on them happens in the Python workers.
//
// The canonical layout is what makes tensor-level dedup work: the same tensors
// under the same names always produce the same bytes, and so the same blob
// digest, whatever file they were read from.
package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

// MaxHeaderSize is the largest JSON header accepted, as in the reference
// implementation.
const MaxHeaderSize = 100_000_000

// metadataKey is the reserved header key for free-form string metadata.
const metadataKey = "__metadata__"

// dtypeSizes maps each supported dtype to its size in bytes.
var dtypeSizes = map[string]int64{
	"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "F8_E8M0": 1,
	"I16": 2, "U16": 2, "F16": 2, "BF16": 2,
	"I32": 4, "U32": 4, "F32": 4,
	"I64": 8, "U64": 8, "F64": 8,
}

// DTypeSize returns the size in bytes of one element of dtype, or false if
// the dtype is not supported.
func DTypeSize(dtype string) (int64, bool) {
	n, ok := dtypeSizes[dtype]
	return n, ok
}

// Tensor is one entry of a safetensors header.
type Tensor struct {
	Name  string
	DType string
	Shape []int64
	// Offset is the absolute byte offset of the tensor's data in the file.
	Offset int64
	// Size is the length of the tensor's data in bytes.
	Size int64
}

// Header is a parsed safetensors header.
type Header struct {
	// Tensors in the order their data appears in the file.
	Tensors []Tensor
	// Metadata is the optional __metadata__ map; nil when absent.
	Metadata map[string]string
	// DataOffset is where the data section starts: 8 + the header length.
	DataOffset int64
}

// Lookup returns the tensor with the given name.
func (h *Header) Lookup(name string) (Tensor, bool) {
	for _, t := range h.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return Tensor{}, false
}

type rawEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// ReadHeader parses and validates the header of a safetensors file of the
// given total size. The data section must be covered exactly by the tensors,
// with no gaps or overlaps, and each tensor's byte length must match its
// dtype and shape.
func ReadHeader(r io.ReaderAt, size int64) (*Header, error) {
	if size < 8 {
		return nil, errors.New("safetensors: file shorter than its 8-byte header length")
	}
	var prefix [8]byte
	if _, err := r.ReadAt(prefix[:], 0); err != nil {
		return nil, fmt.Errorf("safetensors: read header length: %w", err)
	}
	n := binary.LittleEndian.Uint64(prefix[:])
	if n > MaxHeaderSize {
		return nil, fmt.Errorf("safetensors: header of %d bytes exceeds the %d-byte limit", n, MaxHeaderSize)
	}
	if int64(n) > size-8 {
		return nil, fmt.Errorf("safetensors: header of %d bytes runs past the end of a %d-byte file", n, size)
	}
	buf := make([]byte, n)
	if _, err := r.ReadAt(buf, 8); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}
	if len(bytes.TrimSpace(buf)) == 0 || bytes.TrimSpace(buf)[0] != '{' {
		return nil, errors.New("safetensors: header is not a JSON object")
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}
	h := &Header{DataOffset: 8 + int64(n)}
	dataSize := size - h.DataOffset
	for name, msg := range raw {
		if name == metadataKey {
			if err := json.Unmarshal(msg, &h.Metadata); err != nil {
				return nil, fmt.Errorf("safetensors: %s must map strings to strings: %w", metadataKey, err)
			}
			continue
		}
		var e rawEntry
		dec := json.NewDecoder(bytes.NewReader(msg))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&e); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		begin, end := e.DataOffsets[0], e.DataOffsets[1]
		if begin < 0 || end < begin || end > dataSize {
			return nil, fmt.Errorf("safetensors: tensor %q: data offsets [%d, %d] outside a %d-byte data section", name, begin, end, dataSize)
		}
		want, err := ByteSize(e.DType, e.Shape)
		if err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		if end-begin != want {
			return nil, fmt.Errorf("safetensors: tensor %q: %d bytes of data, want %d for %s%v", name, end-begin, want, e.DType, e.Shape)
		}
		if e.Shape == nil {
			e.Shape = []int64{}
		}
		h.Tensors = append(h.Tensors, Tensor{
			Name: name, DType: e.DType, Shape: e.Shape,
			Offset: h.DataOffset + begin, Size: end - begin,
		})
	}

	sort.Slice(h.Tensors, func(i, j int) bool {
		a, b := h.Tensors[i], h.Tensors[j]
		if a.Offset != b.Offset {
			return a.Offset < b.Offset
		}
		return a.Size < b.Size
	})
	next := h.DataOffset
	for _, t := range h.Tensors {
		if t.Offset != next {
			return nil, fmt.Errorf("safetensors: tensor %q starts at data offset %d, want %d (gap or overlap)", t.Name, t.Offset-h.DataOffset, next-h.DataOffset)
		}
		next += t.Size
	}
	if next != size {
		return nil, fmt.Errorf("safetensors: tensors cover %d of %d data bytes", next-h.DataOffset, dataSize)
	}
	return h, nil
}

// ByteSize returns the number of bytes a tensor of the given dtype and shape
// occupies. A nil or empty shape is a scalar.
func ByteSize(dtype string, shape []int64) (int64, error) {
	size, ok := dtypeSizes[dtype]
	if !ok {
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("negative dimension in shape %v", shape)
		}
		if d != 0 && size > math.MaxInt64/d {
			return 0, fmt.Errorf("shape %v overflows", shape)
		}
		size *= d
	}
	return size, nil
}

// File is an open safetensors file.
type File struct {
	*Header
	f *os.File
}

// Open opens a safetensors file and reads its header.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	h, err := ReadHeader(f, info.Size())
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &File{Header: h, f: f}, nil
}

// Data returns a reader over one tensor's bytes.
func (f *File) Data(t Tensor) *io.SectionReader {
	return io.NewSectionReader(f.f, t.Offset, t.Size)
}

// Close closes the file.
func (f *File) Close() error { return f.f.Close() }

// Entry is a tensor to write.
type Entry struct {
	Name  string
	DType string
	Shape []int64
	// Data yields exactly the tensor's bytes.
	Data io.Reader
}

type headerEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// NewReader returns the canonical safetensors encoding of entries and its
// total size. Canonical means: tensors ordered by name, data packed with no
// gaps, a compact JSON header with sorted keys, padded with spaces to a
// multiple of 8 bytes. Metadata is written only when non-empty.
//
// Each entry's Data is read once, in name order, as the returned reader is
// consumed.
func NewReader(entries []Entry, metadata map[string]string) (io.Reader, int64, error) {
	sorted := make([]Entry, len(entries))
	copy(sorted, entries)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })

	header := make(map[string]any, len(sorted)+1)
	if len(metadata) > 0 {
		header[metadataKey] = metadata
	}
	var offset int64
	for i, e := range sorted {
		if e.Name == "" || e.Name == metadataKey {
			return nil, 0, fmt.Errorf("safetensors: invalid tensor name %q", e.Name)
		}
		if i > 0 && sorted[i-1].Name == e.Name {
			return nil, 0, fmt.Errorf("safetensors: duplicate tensor %q", e.Name)
		}
		size, err := ByteSize(e.DType, e.Shape)
		if err != nil {
			return nil, 0, fmt.Errorf("safetensors: tensor %q: %w", e.Name, err)
		}
		shape := e.Shape
		if shape == nil {
			shape = []int64{}
		}
		header[e.Name] = headerEntry{DType: e.DType, Shape: shape, DataOffsets: [2]int64{offset, offset + size}}
		offset += size
	}

	var js bytes.Buffer
	enc := json.NewEncoder(&js)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(header); err != nil {
		return nil, 0, err
	}
	hdr := bytes.TrimRight(js.Bytes(), "\n")
	if pad := (8 - (len(hdr) % 8)) % 8; pad > 0 {
		hdr = append(hdr, strings.Repeat(" ", pad)...)
	}
	prefix := make([]byte, 8, 8+len(hdr))
	binary.LittleEndian.PutUint64(prefix, uint64(len(hdr)))
	head := append(prefix, hdr...)

	readers := []io.Reader{bytes.NewReader(head)}
	for _, e := range sorted {
		size, _ := ByteSize(e.DType, e.Shape)
		readers = append(readers, &exactReader{r: e.Data, left: size, name: e.Name})
	}
	return io.MultiReader(readers...), int64(len(head)) + offset, nil
}

// exactReader fails if its source is shorter or longer than declared, so a
// short read can never produce a well-formed but wrong file.
type exactReader struct {
	r    io.Reader
	left int64
	name string
}

func (e *exactReader) Read(p []byte) (int, error) {
	if e.left == 0 {
		var one [1]byte
		if n, _ := io.ReadFull(e.r, one[:]); n > 0 {
			return 0, fmt.Errorf("safetensors: tensor %q: more data than its shape allows", e.name)
		}
		return 0, io.EOF
	}
	if int64(len(p)) > e.left {
		p = p[:e.left]
	}
	n, err := e.r.Read(p)
	e.left -= int64(n)
	if err == io.EOF {
		if e.left > 0 {
			return n, fmt.Errorf("safetensors: tensor %q: data ends %d bytes short", e.name, e.left)
		}
		err = nil
	}
	return n, err
}
