package weightfile

import (
	"bufio"
	"fmt"
	"io"
	"strings"
	"unicode"
)

// rawLine is one logical line of a Weightfile: comments removed, continuation
// lines joined, split into fields.
type rawLine struct {
	no       int // 1-based line number where the logical line starts
	indented bool
	fields   []string
}

// lex splits a Weightfile into logical lines. A line ending in a backslash
// continues on the next line. A '#' that starts a field begins a comment.
// Double quotes group a field that contains spaces; inside them, \" and \\
// are escapes.
func lex(r io.Reader) ([]rawLine, []error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	var (
		out     []rawLine
		errs    []error
		pending *rawLine
		no      int
	)
	for sc.Scan() {
		no++
		text := sc.Text()
		if no == 1 {
			text = strings.TrimPrefix(text, "\ufeff")
		}
		fields, cont, err := splitFields(text)
		if err != nil {
			errs = append(errs, &Error{Line: no, Msg: err.Error()})
			pending = nil
			continue
		}
		if pending == nil {
			if len(fields) == 0 && !cont {
				continue
			}
			pending = &rawLine{no: no, indented: startsIndented(text)}
		}
		pending.fields = append(pending.fields, fields...)
		if cont {
			continue
		}
		if len(pending.fields) > 0 {
			out = append(out, *pending)
		}
		pending = nil
	}
	if err := sc.Err(); err != nil {
		errs = append(errs, &Error{Line: no, Msg: err.Error()})
	}
	if pending != nil {
		errs = append(errs, &Error{Line: pending.no, Msg: "file ends inside a continued line"})
	}
	return out, errs
}

func startsIndented(s string) bool {
	return s != "" && (s[0] == ' ' || s[0] == '\t')
}

// splitFields splits one physical line into fields. cont reports a trailing
// backslash outside quotes and comments.
func splitFields(s string) (fields []string, cont bool, err error) {
	var (
		cur      strings.Builder
		inField  bool
		inQuote  bool
		runes    = []rune(s)
		flush    = func() { fields = append(fields, cur.String()); cur.Reset(); inField = false }
		lastWasW = true // previous rune was whitespace or start of line
	)
	for i := 0; i < len(runes); i++ {
		c := runes[i]
		switch {
		case inQuote:
			switch c {
			case '\\':
				if i+1 < len(runes) && (runes[i+1] == '"' || runes[i+1] == '\\') {
					i++
					cur.WriteRune(runes[i])
				} else {
					cur.WriteRune(c)
				}
			case '"':
				inQuote = false
			default:
				cur.WriteRune(c)
			}
			lastWasW = false
		case c == '#' && lastWasW:
			if inField {
				flush()
			}
			return fields, false, nil
		case c == '\\' && strings.TrimRightFunc(string(runes[i+1:]), unicode.IsSpace) == "":
			if inField {
				flush()
			}
			return fields, true, nil
		case unicode.IsSpace(c):
			if inField {
				flush()
			}
			lastWasW = true
		case c == '"':
			inQuote, inField, lastWasW = true, true, false
		default:
			cur.WriteRune(c)
			inField, lastWasW = true, false
		}
	}
	if inQuote {
		return nil, false, fmt.Errorf("unterminated quote")
	}
	if inField {
		flush()
	}
	return fields, false, nil
}
