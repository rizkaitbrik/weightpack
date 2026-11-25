package entropy

import (
	"bytes"
	"io"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

func TestZstdRoundTripIsDeterministic(t *testing.T) {
	data := make([]byte, 3<<20)
	rand.New(rand.NewSource(1)).Read(data[:1<<20]) // a random third, the rest zeros
	compress := func() []byte {
		r, err := Encode(Zstd, bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		b, err := io.ReadAll(r)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	a, b := compress(), compress()
	if !bytes.Equal(a, b) {
		t.Fatal("two compressions of the same bytes differ")
	}
	if len(a) >= len(data)/2 {
		t.Errorf("compressed %d bytes to %d", len(data), len(a))
	}
	rc, err := Decode("application/x+zstd", io.NopCloser(bytes.NewReader(a)))
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()
	back, err := io.ReadAll(rc)
	if err != nil || !bytes.Equal(back, data) {
		t.Fatalf("round trip lost data: %v", err)
	}
}

func TestSuffixes(t *testing.T) {
	if Suffix("application/vnd.weightpack.delta.xor.v1+zstd") != "zstd" || Suffix("application/vnd.weightpack.tensor.v1") != "" {
		t.Error("Suffix wrong")
	}
	plain := io.NopCloser(bytes.NewReader([]byte("x")))
	if rc, err := Decode("application/vnd.weightpack.tensor.v1", plain); err != nil || rc != plain {
		t.Error("a plain media type should pass through")
	}
	if _, err := Decode("application/x+gzip", io.NopCloser(bytes.NewReader(nil))); err == nil {
		t.Error("unknown coder accepted")
	}
	if _, err := Encode("gzip", bytes.NewReader(nil)); err == nil {
		t.Error("unknown coder accepted for encoding")
	}
}

// Closing a stream nobody finished reading must stop its encoder goroutine.
func TestCloseStopsTheEncoder(t *testing.T) {
	before := runtime.NumGoroutine()
	for range 20 {
		r, err := Encode(Zstd, bytes.NewReader(make([]byte, 8<<20)))
		if err != nil {
			t.Fatal(err)
		}
		_ = r.Close() // never read
	}
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > before+2 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if n := runtime.NumGoroutine(); n > before+2 {
		t.Errorf("%d goroutines left behind after closing", n-before)
	}
}
