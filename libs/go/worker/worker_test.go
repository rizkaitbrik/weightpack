package worker

import (
	"context"
	"flag"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"

	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// When the test binary runs with WPK_HELPER_WORKER=1 it acts as a worker:
// it serves Info on the socket Spawn passes with --socket.
func TestMain(m *testing.M) {
	if os.Getenv("WPK_HELPER_WORKER") == "1" {
		fs := flag.NewFlagSet("helper", flag.ExitOnError)
		socket := fs.String("socket", "", "")
		_ = fs.Parse(os.Args[len(os.Args)-2:])
		ln, err := net.Listen("unix", *socket)
		if err != nil {
			os.Exit(1)
		}
		srv := grpc.NewServer()
		workerpb.RegisterWorkerServer(srv, helper{})
		_ = srv.Serve(ln)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

type helper struct {
	workerpb.UnimplementedWorkerServer
}

func (helper) Info(context.Context, *workerpb.InfoRequest) (*workerpb.InfoResponse, error) {
	return &workerpb.InfoResponse{Version: "helper"}, nil
}

func shortDir(t *testing.T) string {
	t.Helper()
	d, err := os.MkdirTemp("", "wpkw") // socket paths are limited to ~100 bytes
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(d) })
	return d
}

func TestSpawnAndCall(t *testing.T) {
	t.Setenv("WPK_HELPER_WORKER", "1")
	socket := filepath.Join(shortDir(t), "w.sock")
	p, err := Spawn(context.Background(), []string{os.Args[0], "-test.run=^$"}, socket, io.Discard, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stop()
	c, err := Dial(socket)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	info, err := c.Info(context.Background(), &workerpb.InfoRequest{})
	if err != nil || info.Version != "helper" {
		t.Fatalf("Info = %v, %v", info, err)
	}
	// The helper implements only Info; other calls explain themselves.
	_, err = c.Squash(context.Background(), &workerpb.SquashRequest{})
	if err = Explain(err, socket); err == nil || !strings.HasPrefix(err.Error(), "worker: ") {
		t.Errorf("unimplemented call: %v", err)
	}
}

func TestSpawnReportsAWorkerThatDies(t *testing.T) {
	socket := filepath.Join(shortDir(t), "w.sock")
	_, err := Spawn(context.Background(), []string{"false"}, socket, io.Discard, 10*time.Second)
	if err == nil || !strings.Contains(err.Error(), "exited before it answered") {
		t.Fatalf("err = %v", err)
	}
	if _, err := Spawn(context.Background(), nil, socket, io.Discard, time.Second); err == nil {
		t.Error("empty command accepted")
	}
}

func TestExplainMissingWorker(t *testing.T) {
	socket := filepath.Join(shortDir(t), "absent.sock")
	c, err := Dial(socket)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err = c.Info(ctx, &workerpb.InfoRequest{})
	if err = Explain(err, socket); err == nil || !strings.Contains(err.Error(), "weightpack-worker --socket") {
		t.Errorf("err = %v", err)
	}
}
