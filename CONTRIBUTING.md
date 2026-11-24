# Contributing

The inference server now lives at
**[nonagenticai/mlx-inference-server](https://github.com/nonagenticai/mlx-inference-server)** —
send server changes (Rust, the console, the contract goldens, the launchers) there,
following that repository's `CONTRIBUTING.md`.

This repository holds the Python-side tooling only. For it:

1. Run `python3 scripts/parity_embed_rerank.py --help` to see the harness options.
2. Keep the public-tree rule: never commit internal hostnames, tailnet/CGNAT
   addresses or private repository paths. `scripts/ci/no-private-refs.sh` enforces
   it in CI; the allowlist is `.ci/private-refs.allow`.
3. Open one pull request per change; a green `Tests` run is the gate.

Security reports: see `SECURITY.md`. Do not open a public issue for a vulnerability.
