#!/bin/sh
# Installs Weightpack: wpk and wpkd, then the Python build worker and the
# reference serving engine (weightpack-worker[serve]) with uv.
#
#   curl -fsSL https://raw.githubusercontent.com/rizkaitbrik/weightpack/main/scripts/install.sh | sh
#
# Environment:
#   WPK_VERSION      release to install, such as 0.2.0 (default: the latest)
#   WPK_INSTALL_DIR  where wpk and wpkd go (default: ~/.local/bin)
#   WPK_NO_WORKER=1  skip the Python worker and engine
#   WPK_SERVICE=1    then run `wpk service install`, so wpkd starts at login
#   WPK_RELEASES     releases URL (default: https://github.com/rizkaitbrik/weightpack/releases)
set -eu

releases=${WPK_RELEASES:-https://github.com/rizkaitbrik/weightpack/releases}
dir=${WPK_INSTALL_DIR:-$HOME/.local/bin}

say() { printf '%s\n' "$*"; }
fail() {
	printf 'install: %s\n' "$*" >&2
	exit 1
}

need() { command -v "$1" >/dev/null 2>&1 || fail "$1 is required"; }
need curl
need tar
need uname

os=$(uname -s)
arch=$(uname -m)
case "$arch" in
arm64 | aarch64) arch=arm64 ;;
x86_64 | amd64) arch=amd64 ;;
*) fail "unsupported CPU architecture $arch" ;;
esac
case "$os" in
Darwin)
	os=darwin
	[ "$arch" = arm64 ] || fail "Intel Macs are not supported: torch no longer ships for them. Use an Apple Silicon Mac, Linux, or the container image."
	;;
Linux) os=linux ;;
*) fail "unsupported system $os; Weightpack runs on macOS (Apple Silicon) and Linux" ;;
esac

if [ -n "${WPK_VERSION:-}" ]; then
	version=${WPK_VERSION#v}
else
	# The latest release redirects to its tag page; no API token needed.
	latest=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$releases/latest") || fail "cannot reach $releases"
	version=${latest##*/}
	version=${version#v}
	[ -n "$version" ] && [ "$version" != latest ] || fail "no release found at $releases"
fi

archive="weightpack_${version}_${os}_${arch}.tar.gz"
base="$releases/download/v$version"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

say "Downloading Weightpack $version for $os/$arch"
curl -fsSL -o "$tmp/$archive" "$base/$archive" || fail "cannot download $base/$archive"
curl -fsSL -o "$tmp/checksums.txt" "$base/checksums.txt" || fail "cannot download $base/checksums.txt"

# verify FILE checks a download against checksums.txt.
verify() {
	want=$(awk -v f="$1" '$2 == f { print $1 }' "$tmp/checksums.txt")
	[ -n "$want" ] || fail "checksums.txt does not list $1"
	if command -v sha256sum >/dev/null 2>&1; then
		got=$(sha256sum "$tmp/$1" | awk '{ print $1 }')
	else
		got=$(shasum -a 256 "$tmp/$1" | awk '{ print $1 }')
	fi
	[ "$got" = "$want" ] || fail "checksum mismatch for $1: got $got, want $want"
}
verify "$archive"

tar -xzf "$tmp/$archive" -C "$tmp"
mkdir -p "$dir"
for bin in wpk wpkd; do
	[ -f "$tmp/$bin" ] || fail "$archive has no $bin"
	cp "$tmp/$bin" "$dir/$bin.new"
	chmod 0755 "$dir/$bin.new"
	mv -f "$dir/$bin.new" "$dir/$bin"
done
say "Installed wpk and wpkd in $dir"

if [ "${WPK_NO_WORKER:-}" != 1 ]; then
	wheel="weightpack_worker-$version-py3-none-any.whl"
	if command -v uv >/dev/null 2>&1; then
		say "Installing the build worker and serving engine (weightpack-worker[serve]) with uv"
		curl -fsSL -o "$tmp/$wheel" "$base/$wheel" || fail "cannot download $base/$wheel"
		verify "$wheel"
		uv tool install --force "weightpack-worker[serve] @ file://$tmp/$wheel" || fail "uv tool install failed"
	else
		say ""
		say "uv is not installed, so the build worker and engine were skipped."
		say "Install uv (https://docs.astral.sh/uv/), then run:"
		say "  uv tool install \"weightpack-worker[serve] @ $base/$wheel\""
	fi
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
	say ""
	say "NVIDIA GPU found. For vLLM, which serves many adapters at once: uv tool install vllm"
fi

case ":$PATH:" in
*":$dir:"*) ;;
*)
	say ""
	say "$dir is not on your PATH. Add it, for example:"
	say "  echo 'export PATH=\"$dir:\$PATH\"' >> ~/.profile"
	;;
esac

if [ "${WPK_SERVICE:-}" = 1 ]; then
	"$dir/wpk" service install
else
	say ""
	say "Start the daemon at login with: wpk service install"
fi
say "Then check the machine with: wpk doctor"
