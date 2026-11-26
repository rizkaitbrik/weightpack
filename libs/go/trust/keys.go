package trust

import (
	"context"
	"crypto"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"

	protocommon "github.com/sigstore/protobuf-specs/gen/pb-go/common/v1"
	"github.com/sigstore/sigstore-go/pkg/sign"
	"github.com/sigstore/sigstore/pkg/cryptoutils"
	"github.com/sigstore/sigstore/pkg/signature"
)

// GenerateKeyPair makes an ECDSA P-256 key pair in cosign's formats: the
// private key encrypted with password (scrypt and secretbox, PEM type
// "ENCRYPTED SIGSTORE PRIVATE KEY"), the public key as PKIX PEM. cosign
// reads both, and wpk reads cosign's.
func GenerateKeyPair(password []byte) (privatePEM, publicPEM []byte, err error) {
	return cryptoutils.GeneratePEMEncodedECDSAKeyPair(elliptic.P256(), cryptoutils.StaticPasswordFunc(password))
}

// KeyNeedsPassword reports whether a PEM private key is encrypted.
func KeyNeedsPassword(privatePEM []byte) bool {
	block, _ := pem.Decode(privatePEM)
	if block == nil {
		return false
	}
	switch block.Type {
	case string(cryptoutils.EncryptedSigstorePrivateKeyPEMType), "ENCRYPTED COSIGN PRIVATE KEY", "ENCRYPTED PRIVATE KEY":
		return true
	}
	return false
}

// LoadKey reads a PEM private key, cosign's encrypted format or a plain
// PKCS #8 or SEC 1 one, as a signing key.
func LoadKey(privatePEM, password []byte) (sign.Keypair, error) {
	priv, err := cryptoutils.UnmarshalPEMToPrivateKey(privatePEM, cryptoutils.StaticPasswordFunc(password))
	if err != nil {
		if KeyNeedsPassword(privatePEM) {
			return nil, fmt.Errorf("decrypt the private key (wrong password?): %w", err)
		}
		return nil, fmt.Errorf("read the private key: %w", err)
	}
	signer, ok := priv.(crypto.Signer)
	if !ok {
		return nil, fmt.Errorf("a %T cannot sign", priv)
	}
	alg, err := signature.GetDefaultAlgorithmDetails(signer.Public())
	if err != nil {
		return nil, fmt.Errorf("unsupported signing key: %w", err)
	}
	hint, err := keyHint(signer.Public())
	if err != nil {
		return nil, err
	}
	return &keypair{signer: signer, alg: alg, hint: hint}, nil
}

// ParsePublicKey reads a PEM public key.
func ParsePublicKey(publicPEM []byte) (crypto.PublicKey, error) {
	pub, err := cryptoutils.UnmarshalPEMToPublicKey(publicPEM)
	if err != nil {
		return nil, fmt.Errorf("not a PEM public key: %w", err)
	}
	if _, err := signature.GetDefaultAlgorithmDetails(pub); err != nil {
		return nil, fmt.Errorf("unsupported public key: %w", err)
	}
	return pub, nil
}

// Fingerprint names a public key the way ssh does: SHA256: and the
// unpadded base64 of its PKIX encoding's SHA-256.
func Fingerprint(pub crypto.PublicKey) string {
	der, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		return "unknown key"
	}
	sum := sha256.Sum256(der)
	return "SHA256:" + base64.RawStdEncoding.EncodeToString(sum[:])
}

// keyHint is the hint sigstore-go's ephemeral keys carry: the base64
// SHA-256 of the PKIX public key.
func keyHint(pub crypto.PublicKey) ([]byte, error) {
	der, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(der)
	return []byte(base64.StdEncoding.EncodeToString(sum[:])), nil
}

// keypair is a long-lived key as sigstore-go's sign.Keypair.
type keypair struct {
	signer crypto.Signer
	alg    signature.AlgorithmDetails
	hint   []byte
}

func (k *keypair) GetHashAlgorithm() protocommon.HashAlgorithm { return k.alg.GetProtoHashType() }
func (k *keypair) GetSigningAlgorithm() protocommon.PublicKeyDetails {
	return k.alg.GetSignatureAlgorithm()
}
func (k *keypair) GetHint() []byte                { return k.hint }
func (k *keypair) GetPublicKey() crypto.PublicKey { return k.signer.Public() }

func (k *keypair) GetKeyAlgorithm() string {
	switch k.alg.GetKeyType() {
	case signature.ECDSA:
		return "ECDSA"
	case signature.RSA:
		return "RSA"
	case signature.ED25519:
		return "ED25519"
	}
	return ""
}

func (k *keypair) GetPublicKeyPem() (string, error) {
	b, err := cryptoutils.MarshalPublicKeyToPEM(k.signer.Public())
	return string(b), err
}

// SignData signs data's digest, or data itself for pure Ed25519, as
// sigstore-go's ephemeral keys do.
func (k *keypair) SignData(_ context.Context, data []byte) ([]byte, []byte, error) {
	hf := k.alg.GetHashType()
	toSign := data
	if hf != crypto.Hash(0) {
		h := hf.New()
		h.Write(data)
		toSign = h.Sum(nil)
	}
	sig, err := k.signer.Sign(rand.Reader, toSign, hf)
	if err != nil {
		return nil, nil, err
	}
	return sig, toSign, nil
}

var errNoSignatures = errors.New("no Sigstore signature is attached")
