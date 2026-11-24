package safetensors

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func encode(t *testing.T, entries []Entry, meta map[string]string) []byte {
	t.Helper()
	r, size, err := NewReader(entries, meta)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if int64(len(b)) != size {
		t.Fatalf("size %d, wrote %d bytes", size, len(b))
	}
	return b
}

// rawFile builds a safetensors file from a literal header, for malformed cases.
func rawFile(header string, data []byte) []byte {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, uint64(len(header)))
	b.WriteString(header)
	b.Write(data)
	return b.Bytes()
}

func TestRoundTrip(t *testing.T) {
	entries := []Entry{
		{Name: "b.weight", DType: "BF16", Shape: []int64{2, 3}, Data: bytes.NewReader(bytes.Repeat([]byte{1}, 12))},
		{Name: "a.bias", DType: "F32", Shape: []int64{2}, Data: bytes.NewReader(bytes.Repeat([]byte{2}, 8))},
		{Name: "scalar", DType: "I64", Data: bytes.NewReader(bytes.Repeat([]byte{3}, 8))},
	}
	b := encode(t, entries, map[string]string{"format": "pt"})

	hdrLen := binary.LittleEndian.Uint64(b[:8])
	if (8+hdrLen)%8 != 0 {
		t.Errorf("header not padded to 8 bytes: %d", hdrLen)
	}
	h, err := ReadHeader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	if h.Metadata["format"] != "pt" {
		t.Errorf("metadata = %v", h.Metadata)
	}
	var names []string
	for _, tn := range h.Tensors {
		names = append(names, tn.Name)
	}
	if got := strings.Join(names, ","); got != "a.bias,b.weight,scalar" {
		t.Errorf("data order = %s, want sorted by name", got)
	}
	w, _ := h.Lookup("b.weight")
	if !bytes.Equal(b[w.Offset:w.Offset+w.Size], bytes.Repeat([]byte{1}, 12)) {
		t.Error("b.weight data not at its offset")
	}
	s, _ := h.Lookup("scalar")
	if len(s.Shape) != 0 || s.Size != 8 {
		t.Errorf("scalar = %+v", s)
	}
}

func TestCanonicalIgnoresInputOrder(t *testing.T) {
	mk := func(order []int) []byte {
		all := []Entry{
			{Name: "x", DType: "U8", Shape: []int64{3}},
			{Name: "y", DType: "U8", Shape: []int64{1}},
		}
		data := [][]byte{{1, 2, 3}, {9}}
		var es []Entry
		for _, i := range order {
			e := all[i]
			e.Data = bytes.NewReader(data[i])
			es = append(es, e)
		}
		return encode(t, es, nil)
	}
	if a, b := sha256.Sum256(mk([]int{0, 1})), sha256.Sum256(mk([]int{1, 0})); a != b {
		t.Error("same tensors in a different order gave different bytes")
	}
}

func TestWriterRejectsWrongDataLength(t *testing.T) {
	for name, data := range map[string][]byte{"short": {1, 2}, "long": {1, 2, 3, 4, 5}} {
		r, _, err := NewReader([]Entry{{Name: "x", DType: "U8", Shape: []int64{4}, Data: bytes.NewReader(data)}}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.ReadAll(r); err == nil {
			t.Errorf("%s data: no error", name)
		}
	}
}

func TestWriterRejectsBadEntries(t *testing.T) {
	cases := map[string][]Entry{
		"duplicate": {{Name: "x", DType: "U8", Shape: []int64{1}}, {Name: "x", DType: "U8", Shape: []int64{1}}},
		"dtype":     {{Name: "x", DType: "Q4", Shape: []int64{1}}},
		"reserved":  {{Name: "__metadata__", DType: "U8", Shape: []int64{1}}},
		"negative":  {{Name: "x", DType: "U8", Shape: []int64{-1}}},
	}
	for name, es := range cases {
		if _, _, err := NewReader(es, nil); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestReadHeaderRejectsMalformed(t *testing.T) {
	cases := map[string][]byte{
		"tiny":        {1, 2},
		"not object":  rawFile(`[1]     `, nil),
		"past end":    append(binary.LittleEndian.AppendUint64(nil, 1000), '{', '}'),
		"bad dtype":   rawFile(`{"x":{"dtype":"Q4","shape":[1],"data_offsets":[0,1]}}`, []byte{0}),
		"wrong size":  rawFile(`{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, make([]byte, 4)),
		"gap":         rawFile(`{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}`, make([]byte, 2)),
		"overlap":     rawFile(`{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"y":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}`, make([]byte, 3)),
		"trailing":    rawFile(`{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, make([]byte, 2)),
		"out of file": rawFile(`{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}`, make([]byte, 2)),
		"extra field": rawFile(`{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"q":1}}`, make([]byte, 1)),
		"bad meta":    rawFile(`{"__metadata__":{"a":1}}`, nil),
	}
	for name, b := range cases {
		if _, err := ReadHeader(bytes.NewReader(b), int64(len(b))); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestReadsForeignLayout(t *testing.T) {
	// A file as another writer might lay it out: unsorted data, extra
	// whitespace, metadata. Re-encoding it gives the canonical form.
	header := `{ "z": {"dtype":"U8","shape":[2],"data_offsets":[0,2]}, "a": {"dtype":"U8","shape":[1],"data_offsets":[2,3]}, "__metadata__": {"format":"pt"} }`
	b := rawFile(header, []byte{7, 8, 9})
	path := filepath.Join(t.TempDir(), "m.safetensors")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if f.Tensors[0].Name != "z" {
		t.Errorf("tensors not in file order: %+v", f.Tensors)
	}
	var es []Entry
	for _, tn := range f.Tensors {
		es = append(es, Entry{Name: tn.Name, DType: tn.DType, Shape: tn.Shape, Data: f.Data(tn)})
	}
	out := encode(t, es, nil)
	h, err := ReadHeader(bytes.NewReader(out), int64(len(out)))
	if err != nil {
		t.Fatal(err)
	}
	a, _ := h.Lookup("a")
	z, _ := h.Lookup("z")
	if out[a.Offset] != 9 || out[z.Offset] != 7 || out[z.Offset+1] != 8 {
		t.Errorf("data moved wrongly: %v", out[h.DataOffset:])
	}
}
