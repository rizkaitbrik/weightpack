//go:build !linux

package serving

import (
	"os/exec"
	"syscall"
)

// setProcAttr puts the engine in its own process group, so stopping it
// stops the workers it forks. Where there is no parent-death signal, the
// reference engine exits when its stdin (a pipe from wpkd) closes.
func setProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}
