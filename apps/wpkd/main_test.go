package main

import (
	"context"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strings"
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

func TestServeAPIOverTCP(t *testing.T) {
	root, err := os.MkdirTemp("", "wpkd")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	logger := slog.New(slog.DiscardHandler)

	if err := serve(context.Background(), config{root: root, api: "127.0.0.1:0"}, logger); err == nil || !strings.Contains(err.Error(), "--api-key") {
		t.Fatalf("--api without a key: %v", err)
	}

	addr := freeAddr(t)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- serve(ctx, config{root: root, api: addr, apiToken: "s3cret", gateway: "127.0.0.1:0", maxEngines: 1, node: "gpu-7"}, logger)
	}()
	t.Cleanup(func() {
		cancel()
		<-done
	})

	good := api.NewRemoteClient("http://"+addr, "s3cret", nil)
	var node *api.Node
	deadline := time.Now().Add(5 * time.Second)
	for {
		node, err = good.Node(context.Background())
		if err == nil || time.Now().After(deadline) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		t.Fatal(err)
	}
	if node.Name != "gpu-7" || !node.Serving || node.GatewayPort == 0 || node.MaxEngines != 1 || node.Engines != 0 {
		t.Errorf("node = %+v", node)
	}
	if !slices.Contains(node.Devices, "cpu") || !slices.Contains(node.EngineKinds, "reference") {
		t.Errorf("node devices %v, engines %v", node.Devices, node.EngineKinds)
	}
	if _, err := api.NewRemoteClient("http://"+addr, "wrong", nil).Node(context.Background()); err == nil || !strings.Contains(err.Error(), "API key") {
		t.Errorf("wrong key: %v", err)
	}
	if _, err := api.NewRemoteClient("http://"+addr, "", nil).Images(context.Background()); err == nil {
		t.Error("no key: answered")
	}
}

// freeAddr returns a loopback address with a port nothing listens on.
func freeAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	return addr
}
