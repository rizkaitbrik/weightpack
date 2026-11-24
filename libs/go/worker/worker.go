// Package worker connects wpkd to a Python build worker over gRPC on a Unix
// socket (see proto/weightpack/worker/v1). The worker does all per-weight
// work; the daemon hands it blob paths and stores what it writes.
package worker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// Client is a connection to a worker.
type Client struct {
	workerpb.WorkerClient
	conn   *grpc.ClientConn
	socket string
}

// Dial returns a client for the worker listening on socket. It does not
// connect until the first call, so the worker may start later.
func Dial(socket string) (*Client, error) {
	abs, err := filepath.Abs(socket)
	if err != nil {
		return nil, err
	}
	conn, err := grpc.NewClient("unix://"+abs, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &Client{WorkerClient: workerpb.NewWorkerClient(conn), conn: conn, socket: abs}, nil
}

// Socket returns the socket the client talks to.
func (c *Client) Socket() string { return c.socket }

// Close closes the connection.
func (c *Client) Close() error { return c.conn.Close() }

// Explain turns an error from a worker call into one that says what to do
// when the cause is that no worker is running.
func Explain(err error, socket string) error {
	if status.Code(err) == codes.Unavailable {
		return fmt.Errorf("no build worker answers at %s; start one with `weightpack-worker --socket %s`, or start wpkd with --worker", socket, socket)
	}
	if s, ok := status.FromError(err); ok && s.Code() != codes.OK {
		return fmt.Errorf("worker: %s", s.Message())
	}
	return err
}

// Process is a worker started by Spawn.
type Process struct {
	cmd  *exec.Cmd
	done chan error
}

// Spawn starts a worker with command (a program and its arguments), adding
// --socket socket, and waits until it answers Info. Its output goes to log.
func Spawn(ctx context.Context, command []string, socket string, log io.Writer, timeout time.Duration) (*Process, error) {
	if len(command) == 0 {
		return nil, errors.New("empty worker command")
	}
	args := append(append([]string{}, command[1:]...), "--socket", socket)
	cmd := exec.Command(command[0], args...)
	cmd.Stdout, cmd.Stderr = log, log
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start worker %q: %w", strings.Join(command, " "), err)
	}
	p := &Process{cmd: cmd, done: make(chan error, 1)}
	go func() { p.done <- cmd.Wait() }()

	c, err := Dial(socket)
	if err != nil {
		p.Stop()
		return nil, err
	}
	defer c.Close()
	deadline := time.Now().Add(timeout)
	for {
		callCtx, cancel := context.WithTimeout(ctx, time.Second)
		_, err := c.Info(callCtx, &workerpb.InfoRequest{}, grpc.WaitForReady(true))
		cancel()
		if err == nil {
			return p, nil
		}
		select {
		case werr := <-p.done:
			p.done <- werr
			return nil, fmt.Errorf("worker %q exited before it answered: %v", strings.Join(command, " "), werr)
		default:
		}
		if time.Now().After(deadline) || ctx.Err() != nil {
			p.Stop()
			return nil, fmt.Errorf("worker %q did not answer on %s within %s", strings.Join(command, " "), socket, timeout)
		}
	}
}

// Stop asks the worker to exit and kills it if it has not after 5 seconds.
func (p *Process) Stop() {
	if p.cmd.Process == nil {
		return
	}
	_ = p.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case err := <-p.done:
		p.done <- err
	case <-time.After(5 * time.Second):
		_ = p.cmd.Process.Kill()
	}
}

// DefaultSocket is where wpkd looks for a worker: $WPK_WORKER_SOCKET, or
// <root>/worker.sock.
func DefaultSocket(root string) string {
	if s := os.Getenv("WPK_WORKER_SOCKET"); s != "" {
		return s
	}
	return filepath.Join(root, "worker.sock")
}
