package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/opencontainers/go-digest"
	"github.com/sigstore/sigstore/pkg/oauthflow"
	"golang.org/x/term"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/trust"
)

// passwordEnv holds a signing key's password, as it does for cosign.
const passwordEnv = "COSIGN_PASSWORD"

// signFlags are the flags of every command that signs.
type signFlags struct {
	key   *string
	token *string
	yes   *bool
}

func addSignFlags(fs *flag.FlagSet) signFlags {
	return signFlags{
		key:   fs.String("key", "", "sign with this private key (wpk keygen or cosign generate-key-pair); without it, sign keyless with Sigstore"),
		token: fs.String("identity-token", "", "OIDC token for keyless signing (default: $SIGSTORE_ID_TOKEN, GitHub Actions' token, or a browser login)"),
		yes:   fs.Bool("yes", false, "skip the confirmation that keyless signing logs your identity publicly"),
	}
}

// signer builds the signer the flags ask for. It runs before anything is
// pushed, so a wrong password or a failed login costs nothing.
func (f signFlags) signer(ctx context.Context, stderr io.Writer) (*trust.Signer, error) {
	if *f.key != "" {
		pem, err := os.ReadFile(*f.key)
		if err != nil {
			return nil, err
		}
		var password []byte
		if trust.KeyNeedsPassword(pem) {
			password, err = readPassword(stderr, "Password for "+*f.key+": ", false)
			if err != nil {
				return nil, err
			}
		}
		return trust.NewKeySigner(pem, password)
	}
	sg := &trust.Sigstore{}
	token, err := identityToken(ctx, *f.token, sg, stderr)
	if err != nil {
		return nil, err
	}
	if !*f.yes && term.IsTerminal(int(os.Stdin.Fd())) {
		fmt.Fprintln(stderr, "Keyless signing records your identity and the image digest in Sigstore's public\ntransparency log, where they stay. Sign with --key to keep them private.")
		if !confirm(stderr, "Continue?") {
			return nil, errors.New("not signed")
		}
	}
	return trust.NewKeylessSigner(sg, token)
}

// identityToken finds an OIDC token for keyless signing: the flag, then
// $SIGSTORE_ID_TOKEN, then GitHub Actions' token service, then a browser
// login with the Sigstore instance's identity provider.
func identityToken(ctx context.Context, flagToken string, sg *trust.Sigstore, stderr io.Writer) (string, error) {
	if flagToken != "" {
		return flagToken, nil
	}
	if t := os.Getenv("SIGSTORE_ID_TOKEN"); t != "" {
		return t, nil
	}
	if url, bearer := os.Getenv("ACTIONS_ID_TOKEN_REQUEST_URL"), os.Getenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN"); url != "" && bearer != "" {
		return githubActionsToken(ctx, url, bearer)
	}
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		return "", errors.New("keyless signing needs an OIDC token: pass --identity-token or set SIGSTORE_ID_TOKEN (in GitHub Actions, grant id-token: write), or sign with --key")
	}
	fmt.Fprintln(stderr, "Opening a browser to log in for keyless signing...")
	tok, err := oauthflow.OIDConnect(sg.OIDCIssuer(), "sigstore", "", "", oauthflow.DefaultIDTokenGetter)
	if err != nil {
		return "", fmt.Errorf("log in for keyless signing: %w", err)
	}
	return tok.RawString, nil
}

// githubActionsToken asks GitHub Actions for an ID token with Sigstore's
// audience; the workflow needs the id-token: write permission.
func githubActionsToken(ctx context.Context, url, bearer string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url+"&audience=sigstore", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "bearer "+bearer)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("get GitHub Actions' ID token: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("get GitHub Actions' ID token: %s", resp.Status)
	}
	var body struct {
		Value string `json:"value"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&body); err != nil || body.Value == "" {
		return "", fmt.Errorf("get GitHub Actions' ID token: bad response: %v", err)
	}
	return body.Value, nil
}

// readPassword returns $COSIGN_PASSWORD when it is set, even empty, or
// else asks on the terminal, twice when confirming a new one.
func readPassword(stderr io.Writer, prompt string, confirmNew bool) ([]byte, error) {
	if pw, ok := os.LookupEnv(passwordEnv); ok {
		return []byte(pw), nil
	}
	fd := int(os.Stdin.Fd())
	if !term.IsTerminal(fd) {
		return nil, fmt.Errorf("the key needs a password: set %s", passwordEnv)
	}
	fmt.Fprint(stderr, prompt)
	pw, err := term.ReadPassword(fd)
	fmt.Fprintln(stderr)
	if err != nil {
		return nil, err
	}
	if confirmNew {
		fmt.Fprint(stderr, "Again: ")
		again, err := term.ReadPassword(fd)
		fmt.Fprintln(stderr)
		if err != nil {
			return nil, err
		}
		if string(again) != string(pw) {
			return nil, errors.New("the passwords differ")
		}
	}
	return pw, nil
}

func confirm(stderr io.Writer, question string) bool {
	fmt.Fprintf(stderr, "%s [y/N] ", question)
	line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
	answer := strings.ToLower(strings.TrimSpace(line))
	return answer == "y" || answer == "yes"
}

// signRemote signs the manifest d in ref's repository and stores the
// signature there.
func signRemote(ctx context.Context, signer *trust.Signer, ref, d string, plainHTTP bool) (*api.AttachResponse, error) {
	bundle, err := signer.Sign(ctx, digest.Digest(d))
	if err != nil {
		return nil, err
	}
	return client().Attach(ctx, api.AttachRequest{Ref: ref, Digest: d, Bundle: bundle, PlainHTTP: plainHTTP})
}

func printSigned(stdout io.Writer, res *api.AttachResponse, who string) {
	fmt.Fprintf(stdout, "Signed %s as %s (signature %s)\n", res.Ref, who, res.Signature)
}

func runSign(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("sign", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sf := addSignFlags(fs)
	plain := fs.Bool("plain-http", false, "talk to the registry over HTTP (always on for localhost and 127.0.0.1)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk sign [--key FILE] [--plain-http] <registry/name[:tag] | registry/name@sha256:…>\n\nSigns an image already in a registry and stores the signature next to it,\nas cosign sign does. The tag resolves to a digest first; the signature\ncovers that manifest and so every layer it names.")
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
	signer, err := sf.signer(ctx, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "wpk sign: %v\n", err)
		return 1
	}
	target, err := client().Resolve(ctx, api.ResolveRequest{Ref: pos[0], PlainHTTP: *plain})
	if err != nil {
		fmt.Fprintf(stderr, "wpk sign: %v\n", err)
		return 1
	}
	res, err := signRemote(ctx, signer, pos[0], target.Digest, *plain)
	if err != nil {
		fmt.Fprintf(stderr, "wpk sign: %v\n", err)
		return 1
	}
	printSigned(stdout, res, signer.Who)
	return 0
}

func runVerify(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var keys, subjects, subjectRes stringList
	fs.Var(&keys, "key", "trust this public key file (repeatable)")
	fs.Var(&subjects, "identity", "trust this keyless signer: an email or a workflow URI (repeatable)")
	fs.Var(&subjectRes, "identity-regexp", "trust keyless signers matching this regular expression, whole (repeatable)")
	issuer := fs.String("issuer", "", "the OIDC issuer of --identity signers, such as https://token.actions.githubusercontent.com")
	issuerRe := fs.String("issuer-regexp", "", "a regular expression the issuer must match, whole")
	plain := fs.Bool("plain-http", false, "talk to the registry over HTTP (always on for localhost and 127.0.0.1)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: wpk verify [--key FILE]... [--identity ID --issuer URL]... [--plain-http] <registry/name[:tag] | …@sha256:…>\n\nChecks an image's Sigstore signatures in its registry. With no --key or\n--identity, checks it against the trust policy wpk pull applies.")
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
	req := api.VerifyRequest{Ref: pos[0], PlainHTTP: *plain}
	if req.Keys, err = readKeys(keys); err != nil {
		fmt.Fprintf(stderr, "wpk verify: %v\n", err)
		return 1
	}
	if req.Identities, err = identities(subjects, subjectRes, *issuer, *issuerRe); err != nil {
		fmt.Fprintf(stderr, "wpk verify: %v\n", err)
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res, err := client().Verify(ctx, req)
	if err != nil {
		fmt.Fprintf(stderr, "wpk verify: %v\n", err)
		return 1
	}
	at := res.Ref
	if !strings.Contains(at, "@") {
		at += "@" + res.Digest
	}
	switch {
	case res.Signer != "":
		fmt.Fprintf(stdout, "Verified %s: signed by %s", at, res.Signer)
		if res.Rule != "" {
			fmt.Fprintf(stdout, " (trust rule %s)", res.Rule)
		}
		fmt.Fprintln(stdout)
	default:
		rule := "no trust rule matches it"
		if res.Rule != "" {
			rule = "trust rule " + res.Rule
		}
		fmt.Fprintf(stdout, "%s: wpk pull takes it unsigned (%s); %d signatures attached\n", at, rule, res.Signatures)
	}
	return 0
}

func readKeys(paths []string) ([]string, error) {
	var out []string
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		if _, err := trust.ParsePublicKey(raw); err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		out = append(out, string(raw))
	}
	return out, nil
}

// identities pairs each subject with the issuer flags.
func identities(subjects, subjectRes []string, issuer, issuerRe string) ([]api.Identity, error) {
	if len(subjects)+len(subjectRes) == 0 {
		if issuer != "" || issuerRe != "" {
			return nil, errors.New("--issuer goes with --identity or --identity-regexp")
		}
		return nil, nil
	}
	if (issuer == "") == (issuerRe == "") {
		return nil, errors.New("keyless identities need exactly one of --issuer and --issuer-regexp")
	}
	var out []api.Identity
	for _, s := range subjects {
		out = append(out, api.Identity{Subject: s, Issuer: issuer, IssuerRegexp: issuerRe})
	}
	for _, s := range subjectRes {
		out = append(out, api.Identity{SubjectRegexp: s, Issuer: issuer, IssuerRegexp: issuerRe})
	}
	return out, nil
}

func runKeygen(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("keygen", flag.ContinueOnError)
	fs.SetOutput(stderr)
	prefix := fs.String("o", "weightpack", "write PREFIX.key and PREFIX.pub")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: wpk keygen [-o PREFIX]\n\nWrites a signing key pair in cosign's format: PREFIX.key, encrypted with a\npassword (from $%s, or asked for), and PREFIX.pub, for wpk trust\nand wpk verify. Keep the .key secret; share the .pub.\n", passwordEnv)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fs.Usage()
		return 2
	}
	keyPath, pubPath := *prefix+".key", *prefix+".pub"
	for _, p := range []string{keyPath, pubPath} {
		if _, err := os.Stat(p); err == nil {
			fmt.Fprintf(stderr, "wpk keygen: %s exists; remove it or pick another -o\n", p)
			return 1
		}
	}
	pw, err := readPassword(stderr, "Password for the new key: ", true)
	if err != nil {
		fmt.Fprintf(stderr, "wpk keygen: %v\n", err)
		return 1
	}
	priv, pub, err := trust.GenerateKeyPair(pw)
	if err != nil {
		fmt.Fprintf(stderr, "wpk keygen: %v\n", err)
		return 1
	}
	if err := writeNew(keyPath, priv, 0o600); err != nil {
		fmt.Fprintf(stderr, "wpk keygen: %v\n", err)
		return 1
	}
	if err := writeNew(pubPath, pub, 0o644); err != nil {
		fmt.Fprintf(stderr, "wpk keygen: %v\n", err)
		return 1
	}
	pk, _ := trust.ParsePublicKey(pub)
	fmt.Fprintf(stdout, "Wrote %s (private) and %s (public), key %s\n", keyPath, pubPath, trust.Fingerprint(pk))
	return 0
}

// writeNew writes a file that must not exist yet.
func writeNew(path string, data []byte, mode os.FileMode) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

func runTrust(args []string, stdout, stderr io.Writer) int {
	usage := func() {
		fmt.Fprintln(stderr, `usage: wpk trust [ls]
       wpk trust set <scope> (--key FILE... | --identity ID... --issuer URL | --accept | --reject)
       wpk trust rm <scope>

The trust policy says which images wpk pull (and builds FROM a registry)
take. The rule whose scope names an image most closely applies: a registry
host (ghcr.io), a repository or a prefix of one (ghcr.io/acme), or * for
every image. Without a matching rule, wpkd pulls the image.`)
	}
	if len(args) == 0 || args[0] == "ls" {
		return trustList(stdout, stderr)
	}
	switch args[0] {
	case "set":
		return trustSet(args[1:], stdout, stderr)
	case "rm":
		if len(args) != 2 {
			usage()
			return 2
		}
		return trustRemove(args[1], stdout, stderr)
	case "-h", "--help", "help":
		usage()
		return 0
	}
	usage()
	return 2
}

func trustList(stdout, stderr io.Writer) int {
	p, err := client().Trust(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "wpk trust: %v\n", err)
		return 1
	}
	if len(p.Rules) == 0 {
		fmt.Fprintln(stdout, "No trust rules: wpk pull takes every image, signed or not. Require signatures with wpk trust set.")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "SCOPE\tACTION\tSIGNERS")
	for _, r := range p.Rules {
		var signers []string
		for _, k := range r.Keys {
			if pub, err := trust.ParsePublicKey([]byte(k)); err == nil {
				signers = append(signers, "key "+trust.Fingerprint(pub))
			}
		}
		for _, id := range r.Identities {
			signers = append(signers, id.String())
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", r.Scope, r.Action, strings.Join(signers, ", "))
	}
	_ = tw.Flush()
	return 0
}

func trustSet(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("trust set", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var keys, subjects, subjectRes stringList
	fs.Var(&keys, "key", "require a signature by this public key file (repeatable)")
	fs.Var(&subjects, "identity", "require a keyless signature by this signer (repeatable)")
	fs.Var(&subjectRes, "identity-regexp", "require a keyless signer matching this regular expression, whole (repeatable)")
	issuer := fs.String("issuer", "", "the OIDC issuer of the keyless signers")
	issuerRe := fs.String("issuer-regexp", "", "a regular expression the issuer must match, whole")
	accept := fs.Bool("accept", false, "take images under the scope signed or not")
	reject := fs.Bool("reject", false, "refuse every image under the scope")
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		return 2
	}
	if len(pos) != 1 {
		fmt.Fprintln(stderr, "usage: wpk trust set <scope> (--key FILE... | --identity ID... --issuer URL | --accept | --reject)")
		return 2
	}
	rule := api.TrustRule{Scope: pos[0]}
	if rule.Keys, err = readKeys(keys); err != nil {
		fmt.Fprintf(stderr, "wpk trust: %v\n", err)
		return 1
	}
	if rule.Identities, err = identities(subjects, subjectRes, *issuer, *issuerRe); err != nil {
		fmt.Fprintf(stderr, "wpk trust: %v\n", err)
		return 2
	}
	signed := len(rule.Keys)+len(rule.Identities) > 0
	switch {
	case btoi(signed)+btoi(*accept)+btoi(*reject) != 1:
		fmt.Fprintln(stderr, "wpk trust: give signers (--key, --identity), or --accept, or --reject")
		return 2
	case signed:
		rule.Action = api.TrustSigned
	case *accept:
		rule.Action = api.TrustAccept
	default:
		rule.Action = api.TrustReject
	}
	scope, err := trust.NormalizeScope(rule.Scope)
	if err != nil {
		fmt.Fprintf(stderr, "wpk trust: %v\n", err)
		return 2
	}
	rule.Scope = scope
	ctx := context.Background()
	p, err := client().Trust(ctx)
	if err != nil {
		fmt.Fprintf(stderr, "wpk trust: %v\n", err)
		return 1
	}
	replaced := false
	for i := range p.Rules {
		if p.Rules[i].Scope == scope {
			p.Rules[i], replaced = rule, true
		}
	}
	if !replaced {
		p.Rules = append(p.Rules, rule)
	}
	if _, err := client().SetTrust(ctx, *p); err != nil {
		fmt.Fprintf(stderr, "wpk trust: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Trust rule for %s: %s\n", scope, rule.Action)
	return 0
}

func trustRemove(scope string, stdout, stderr io.Writer) int {
	norm, err := trust.NormalizeScope(scope)
	if err != nil {
		fmt.Fprintf(stderr, "wpk trust: %v\n", err)
		return 2
	}
	ctx := context.Background()
	p, err := client().Trust(ctx)
	if err != nil {
		fmt.Fprintf(stderr, "wpk trust: %v\n", err)
		return 1
	}
	kept := p.Rules[:0]
	for _, r := range p.Rules {
		if r.Scope != norm {
			kept = append(kept, r)
		}
	}
	if len(kept) == len(p.Rules) {
		fmt.Fprintf(stderr, "wpk trust: no rule for %s\n", norm)
		return 1
	}
	p.Rules = kept
	if _, err := client().SetTrust(ctx, *p); err != nil {
		fmt.Fprintf(stderr, "wpk trust: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "Removed the trust rule for %s\n", norm)
	return 0
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
