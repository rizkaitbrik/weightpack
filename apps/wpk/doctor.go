package main

import (
	"context"
	"fmt"
	"io"
	"text/tabwriter"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

var checkMarks = map[string]string{api.CheckOK: "ok", api.CheckWarn: "warn", api.CheckFail: "FAIL", api.CheckSkip: "-"}

func runDoctor(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		fmt.Fprintln(stderr, "usage: wpk doctor\n\nChecks wpkd, its build worker, the device and engines it would serve on,\nthe gateway, disk space and Hugging Face access, and says how to fix what\nfails.")
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	report, err := client().Doctor(ctx)
	if err != nil {
		report = &api.DoctorReport{Checks: []api.Check{{
			Name: "wpkd", Status: api.CheckFail, Detail: err.Error(),
			Fix: "start it: wpkd --worker weightpack-worker, or install it as a service: wpk service install",
		}}}
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	failed := false
	for _, c := range report.Checks {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", checkMarks[c.Status], c.Name, c.Detail)
		if c.Fix != "" {
			fmt.Fprintf(tw, "\t\tfix: %s\n", c.Fix)
		}
		failed = failed || c.Status == api.CheckFail
	}
	_ = tw.Flush()
	if failed {
		return 1
	}
	return 0
}
