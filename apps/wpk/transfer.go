package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/trust"
)

func runPush(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("push", flag.ContinueOnError)
	fs.SetOutput(stderr)
	plain := fs.Bool("plain-http", false, "talk to the registry over HTTP (always on for localhost and 127.0.0.1)")
	sign := fs.Bool("sign", false, "sign the pushed image (with --key, or keyless with Sigstore)")
	sf := addSignFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk push [--sign [--key FILE]] [--plain-http] <registry/name[:tag]>\n\nSends only the blobs the registry lacks. Credentials come from docker login.\nWith --sign, also signs the image, as wpk sign does.")
		fs.PrintDefaults()
	}
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		return 2
	}
	if len(pos) != 1 {
		fs.Usage()
		return 2
	}
	if !*sign && (*sf.key != "" || *sf.token != "") {
		fmt.Fprintln(stderr, "wpk push: --key and --identity-token go with --sign")
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var signer *trust.Signer
	if *sign {
		if signer, err = sf.signer(ctx, stderr); err != nil {
			fmt.Fprintf(stderr, "wpk push: %v\n", err)
			return 1
		}
	}
	res, err := client().Push(ctx, api.PushRequest{Ref: pos[0], PlainHTTP: *plain}, printEvent(stdout, stderr))
	if err != nil {
		fmt.Fprintf(stderr, "wpk push: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Pushed %s (%s): sent %d of %d blobs, %s of %s\n", res.Ref, res.Digest,
		res.Moved, res.Blobs, api.FormatSize(res.MovedBytes), api.FormatSize(res.Size))
	if signer != nil {
		signed, err := signRemote(ctx, signer, res.Ref, res.Digest, *plain)
		if err != nil {
			fmt.Fprintf(stderr, "wpk push: pushed, but not signed: %v\n", err)
			return 1
		}
		printSigned(stdout, signed, signer.Who)
	}
	return 0
}

func runPull(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	fs.SetOutput(stderr)
	plain := fs.Bool("plain-http", false, "talk to the registry over HTTP (always on for localhost and 127.0.0.1)")
	exact := fs.Bool("exact", false, "refuse images with lossy layers")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk pull [--exact] [--plain-http] <registry/name[:tag] | registry/name@sha256:…>\n\nFetches only the blobs the local store lacks, after checking the image's\nsignatures against the trust policy (wpk trust) and its manifest and config.")
		fs.PrintDefaults()
	}
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		return 2
	}
	if len(pos) != 1 {
		fs.Usage()
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res, err := client().Pull(ctx, api.PullRequest{Ref: pos[0], PlainHTTP: *plain, Exact: *exact}, printEvent(stdout, stderr))
	if err != nil {
		fmt.Fprintf(stderr, "wpk pull: %v\n", err)
		return 1
	}
	if res.Verified != "" {
		fmt.Fprintf(stdout, "Verified %s: signed by %s\n", res.Digest, res.Verified)
	}
	fmt.Fprintf(stdout, "Pulled %s: fetched %d of %d blobs, %s of %s\n", res.Digest,
		res.Moved, res.Blobs, api.FormatSize(res.MovedBytes), api.FormatSize(res.Size))
	if res.Tag != "" {
		fmt.Fprintf(stdout, "Tagged %s\n", res.Tag)
	}
	return 0
}

func runTag(args []string, stdout, stderr io.Writer) int {
	if len(args) != 2 {
		fmt.Fprintln(stderr, "usage: wpk tag <source ref | digest> <target name[:tag]>")
		return 2
	}
	res, err := client().Tag(context.Background(), api.TagRequest{Source: args[0], Target: args[1]})
	if err != nil {
		fmt.Fprintf(stderr, "wpk tag: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Tagged %s (%s)\n", res.Ref, res.Digest)
	return 0
}

// printEvent prints progress lines to stdout and warnings to stderr.
func printEvent(stdout, stderr io.Writer) func(api.Event) {
	return func(e api.Event) {
		if e.Warning {
			fmt.Fprintf(stderr, "warning: %s: %s\n", e.Step, e.Message)
			return
		}
		fmt.Fprintf(stdout, "%-18s %s\n", e.Step, e.Message)
	}
}
