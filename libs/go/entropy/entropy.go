// Package entropy implements the entropy coders a blob's media type names as
// a suffix, as OCI does with +gzip: application/vnd.weightpack.delta.xor.v1
// +zstd is a delta.xor.v1 blob compressed with zstd.
//
// Compression is byte-level, like hashing, so the daemon does it: it
// compresses what it stores and decompresses what it hands to a worker, and
// workers only ever see plain safetensors.
package entropy

import (
	"fmt"
	"io"
	"strings"

	"github.com/klauspost/compress/zstd"
)

// Zstd is the suffix for zstd-compressed blobs.
const Zstd = "zstd"

// Suffix returns the entropy coder a media type names, or "" for none.
func Suffix(mediaType string) string {
	_, coder, _ := strings.Cut(mediaType, "+")
	return coder
}

// Encode returns a reader of r's bytes compressed with coder. The output
// depends only on the input (one encoder goroutine, a fixed level), so the
// same tensor always gives the same blob digest with a given encoder
// version.
//
// The caller must Close the reader, even after reading it to the end:
// closing early stops the encoder goroutine, which would otherwise block on
// its pipe forever.
func Encode(coder string, r io.Reader) (io.ReadCloser, error) {
	if coder != Zstd {
		return nil, fmt.Errorf("unsupported entropy coder %q", coder)
	}
	pr, pw := io.Pipe()
	go func() {
		enc, err := zstd.NewWriter(pw, zstd.WithEncoderLevel(zstd.SpeedDefault), zstd.WithEncoderConcurrency(1))
		if err != nil {
			pw.CloseWithError(err)
			return
		}
		if _, err := io.Copy(enc, r); err != nil {
			_ = enc.Close()
			pw.CloseWithError(err)
			return
		}
		pw.CloseWithError(enc.Close())
	}()
	return pr, nil
}

// Decode returns a reader of a blob's plain bytes, given the media type it is
// stored under. Closing it closes rc.
func Decode(mediaType string, rc io.ReadCloser) (io.ReadCloser, error) {
	switch Suffix(mediaType) {
	case "":
		return rc, nil
	case Zstd:
		dec, err := zstd.NewReader(rc, zstd.WithDecoderConcurrency(1))
		if err != nil {
			_ = rc.Close()
			return nil, err
		}
		return &decoder{dec: dec, rc: rc}, nil
	default:
		_ = rc.Close()
		return nil, fmt.Errorf("unsupported entropy coder in media type %q", mediaType)
	}
}

type decoder struct {
	dec *zstd.Decoder
	rc  io.ReadCloser
}

func (d *decoder) Read(p []byte) (int, error) { return d.dec.Read(p) }

func (d *decoder) Close() error {
	d.dec.Close()
	return d.rc.Close()
}
