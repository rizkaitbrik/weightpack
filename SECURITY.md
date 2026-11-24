# Security policy

Weightpack moves model weights between machines, so a malicious layer is a real threat: a delta can carry a backdoor as easily as a skill.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's [private vulnerability reporting](https://github.com/rizkaitbrik/weightpack/security/advisories/new) for this repository. You will get an answer within a week.

## Scope

- Signature or digest checks that can be bypassed on pull.
- Code execution while loading or decoding a layer (Weightpack loads safetensors only; anything else is a bug).
- Path traversal or unsafe writes in the local blob store.

The project is pre-release; there are no supported versions yet.
