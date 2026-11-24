package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
)

// DefaultRoot is the daemon's store directory: $WPK_ROOT, or ~/.weightpack.
func DefaultRoot() string {
	if dir := os.Getenv("WPK_ROOT"); dir != "" {
		return dir
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".weightpack")
	}
	return ".weightpack"
}

// DefaultSocket is the daemon's socket: $WPK_SOCKET, or <root>/wpkd.sock.
func DefaultSocket() string {
	if s := os.Getenv("WPK_SOCKET"); s != "" {
		return s
	}
	return filepath.Join(DefaultRoot(), "wpkd.sock")
}

// ErrorBody is the JSON body of every non-2xx response.
type ErrorBody struct {
	Error string `json:"error"`
}

// Client talks to wpkd over its Unix socket.
type Client struct {
	socket string
	hc     *http.Client
}

// NewClient returns a client for the daemon listening on socket.
func NewClient(socket string) *Client {
	tr := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", socket)
		},
	}
	return &Client{socket: socket, hc: &http.Client{Transport: tr}}
}

func (c *Client) do(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, "http://wpkd"+path, rd)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		var opErr *net.OpError
		if errors.As(err, &opErr) && opErr.Op == "dial" {
			return nil, fmt.Errorf("cannot reach wpkd at %s; is the daemon running? (start it with: wpkd)", c.socket)
		}
		return nil, err
	}
	if resp.StatusCode >= 300 {
		defer resp.Body.Close()
		var eb ErrorBody
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if json.Unmarshal(raw, &eb) == nil && eb.Error != "" {
			return nil, errors.New(eb.Error)
		}
		return nil, fmt.Errorf("wpkd: %s", resp.Status)
	}
	return resp, nil
}

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	resp, err := c.do(ctx, http.MethodGet, path, nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(out)
}

// Health returns the daemon's health document.
func (c *Client) Health(ctx context.Context) (map[string]string, error) {
	var h map[string]string
	return h, c.getJSON(ctx, "/healthz", &h)
}

// Images lists the tagged images in the daemon's store.
func (c *Client) Images(ctx context.Context) ([]Image, error) {
	var out []Image
	return out, c.getJSON(ctx, "/v1/images", &out)
}

// Inspect returns an image's manifest and config.
func (c *Client) Inspect(ctx context.Context, ref string) (*ImageDetail, error) {
	var out ImageDetail
	if err := c.getJSON(ctx, "/v1/images/inspect?ref="+url.QueryEscape(ref), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Build runs a build and calls onEvent for each progress event as it
// arrives. It returns the result, or the build's error.
func (c *Client) Build(ctx context.Context, req BuildRequest, onEvent func(Event)) (*BuildResult, error) {
	last, err := c.stream(ctx, "/v1/build", req, onEvent)
	if err != nil {
		return nil, err
	}
	if last.Result == nil {
		return nil, errors.New("wpkd ended the build stream without a result")
	}
	return last.Result, nil
}

// Push sends a local image to its registry, reporting progress to onEvent.
func (c *Client) Push(ctx context.Context, req PushRequest, onEvent func(Event)) (*TransferResult, error) {
	return c.transfer(ctx, "/v1/push", req, onEvent)
}

// Pull fetches an image into the daemon's store, reporting progress to
// onEvent.
func (c *Client) Pull(ctx context.Context, req PullRequest, onEvent func(Event)) (*TransferResult, error) {
	return c.transfer(ctx, "/v1/pull", req, onEvent)
}

// Tag points req.Target at the image req.Source names.
func (c *Client) Tag(ctx context.Context, req TagRequest) (*TagResponse, error) {
	resp, err := c.do(ctx, http.MethodPost, "/v1/tag", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out TagResponse
	return &out, json.NewDecoder(resp.Body).Decode(&out)
}

func (c *Client) transfer(ctx context.Context, path string, req any, onEvent func(Event)) (*TransferResult, error) {
	last, err := c.stream(ctx, path, req, onEvent)
	if err != nil {
		return nil, err
	}
	if last.Transfer == nil {
		return nil, errors.New("wpkd ended the stream without a result")
	}
	return last.Transfer, nil
}

// stream posts req and reads the NDJSON reply, passing progress events to
// onEvent. It returns the final event, which carries a result, or the error
// that final event reports.
func (c *Client) stream(ctx context.Context, path string, req any, onEvent func(Event)) (Event, error) {
	resp, err := c.do(ctx, http.MethodPost, path, req)
	if err != nil {
		return Event{}, err
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 64<<20)
	for sc.Scan() {
		var e Event
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			return Event{}, fmt.Errorf("bad event from wpkd: %w", err)
		}
		switch {
		case e.Error != "":
			return Event{}, errors.New(e.Error)
		case e.Result != nil || e.Transfer != nil:
			return e, nil
		case onEvent != nil:
			onEvent(e)
		}
	}
	if err := sc.Err(); err != nil {
		return Event{}, err
	}
	return Event{}, errors.New("wpkd closed the stream without a result")
}
