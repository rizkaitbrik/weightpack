package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

func TestHealthz(t *testing.T) {
	s, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	newMux(s).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
}

func TestBlobLookup(t *testing.T) {
	s, _ := store.Open(t.TempDir())
	d, _, err := s.Put(bytes.NewReader([]byte("blob")), "")
	if err != nil {
		t.Fatal(err)
	}
	mux := newMux(s)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/blobs/"+d, nil))
	if rec.Code != http.StatusNoContent {
		t.Fatalf("present blob: status %d", rec.Code)
	}

	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/blobs/sha256:"+string(bytes.Repeat([]byte("0"), 64)), nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("missing blob: status %d", rec.Code)
	}

	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/blobs/bogus", nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad digest: status %d", rec.Code)
	}
}
