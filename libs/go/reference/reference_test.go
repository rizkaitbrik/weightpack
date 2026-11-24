package reference

import (
	"strings"
	"testing"
)

func TestParse(t *testing.T) {
	d := "sha256:" + strings.Repeat("ab", 32)
	cases := []struct {
		in   string
		want Reference
	}{
		{"llama", Reference{Path: "llama"}},
		{"acme/llama:3.1", Reference{Path: "acme/llama", Tag: "3.1"}},
		{"registry.example.com/llama@" + d, Reference{Host: "registry.example.com", Path: "llama", Digest: d}},
		{"localhost:5000/a/b:v1@" + d, Reference{Host: "localhost:5000", Path: "a/b", Tag: "v1", Digest: d}},
		{"localhost/x", Reference{Host: "localhost", Path: "x"}},
	}
	for _, c := range cases {
		got, err := Parse(c.in)
		if err != nil {
			t.Errorf("Parse(%q): %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("Parse(%q) = %+v, want %+v", c.in, got, c.want)
		}
		if got.String() != c.in {
			t.Errorf("String() = %q, want %q", got.String(), c.in)
		}
	}
}

func TestParseRejects(t *testing.T) {
	for in, msg := range map[string]string{
		"":                "missing repository",
		"UPPER":           "bad path component",
		"a//b":            "bad path component",
		"r.io/a:bad tag":  "bad tag",
		"r.io:port/x":     "bad registry host",
		"x@sha256:abc":    "digest must be",
		"x@md5:" + "00":   "digest must be",
		"-leading/dash":   "bad path component",
		"trailing/dash-/": "bad path component",
	} {
		_, err := Parse(in)
		if err == nil || !strings.Contains(err.Error(), msg) {
			t.Errorf("Parse(%q) = %v, want error containing %q", in, err, msg)
		}
	}
}

func TestParseNamed(t *testing.T) {
	r, err := ParseNamed("assistant")
	if err != nil || r.String() != "assistant:latest" {
		t.Errorf("ParseNamed(assistant) = %v, %v", r, err)
	}
	if _, err := ParseNamed("a@sha256:" + strings.Repeat("0", 64)); err == nil {
		t.Error("digest accepted as a tag")
	}
	if !IsDigest("sha256:"+strings.Repeat("0", 64)) || IsDigest("sha256:0") {
		t.Error("IsDigest wrong")
	}
}
