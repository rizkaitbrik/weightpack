package main

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

func TestServeOnPrivateSocket(t *testing.T) {
	// Unix socket paths are limited to about 100 bytes, so avoid t.TempDir.
	root, err := os.MkdirTemp("", "wpkd")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	socket := filepath.Join(root, "wpkd.sock")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- serve(ctx, config{root: root, socket: socket}, slog.New(slog.DiscardHandler)) }()

	c := api.NewClient(socket)
	var health map[string]string
	deadline := time.Now().Add(5 * time.Second)
	for {
		health, err = c.Health(context.Background())
		if err == nil || time.Now().After(deadline) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil || health["status"] != "ok" {
		t.Fatalf("health = %v, %v", health, err)
	}
	info, err := os.Stat(socket)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("socket mode %o, want 600", perm)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("serve: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("serve did not stop")
	}
}
