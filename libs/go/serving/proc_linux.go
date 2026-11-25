package serving

import (
	"os/exec"
	"syscall"
)

// setProcAttr puts the engine in its own process group, so stopping it
// stops the workers it forks, and has the kernel stop it if wpkd dies.
func setProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGTERM}
}
