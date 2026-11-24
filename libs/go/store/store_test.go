package store

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"testing"
)

func digestOf(b []byte) string {
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func TestPutHasOpenDelete(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	data := []byte("tensor bytes")
	d, n, err := s.Put(bytes.NewReader(data), "")
	if err != nil {
		t.Fatal(err)
	}
	if d != digestOf(data) || n != int64(len(data)) {
		t.Fatalf("Put = %s %d", d, n)
	}
	if ok, _ := s.Has(d); !ok {
		t.Fatal("blob missing after Put")
	}

	// Writing the same content again is a no-op that returns the same digest.
	d2, _, err := s.Put(bytes.NewReader(data), d)
	if err != nil || d2 != d {
		t.Fatalf("second Put = %s, %v", d2, err)
	}

	r, err := s.Open(d)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(r)
	_ = r.Close()
	if !bytes.Equal(got, data) {
		t.Fatalf("Open returned %q", got)
	}

	if err := s.Delete(d); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Open(d); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Open after Delete: %v", err)
	}
}

func TestPutRejectsWrongDigest(t *testing.T) {
	s, _ := Open(t.TempDir())
	wrong := digestOf([]byte("something else"))
	_, _, err := s.Put(bytes.NewReader([]byte("tensor bytes")), wrong)
	if !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("want ErrDigestMismatch, got %v", err)
	}
	if ok, _ := s.Has(wrong); ok {
		t.Fatal("mismatched blob was stored")
	}
}

func TestMissing(t *testing.T) {
	s, _ := Open(t.TempDir())
	have, _, _ := s.Put(bytes.NewReader([]byte("a")), "")
	lack := digestOf([]byte("b"))
	missing, err := s.Missing([]string{have, lack})
	if err != nil {
		t.Fatal(err)
	}
	if len(missing) != 1 || missing[0] != lack {
		t.Fatalf("Missing = %v", missing)
	}
}

func TestParseDigest(t *testing.T) {
	for _, bad := range []string{"", "md5:abc", "sha256:xyz", "sha256:" + string(make([]byte, 64))} {
		if _, err := ParseDigest(bad); err == nil {
			t.Errorf("ParseDigest(%q) accepted", bad)
		}
	}
}
