# Contributing to MLX Inference

Thanks for your interest. Contributions of every size are welcome — bug reports,
documentation fixes, and code.

By participating you agree to the
[Code of Conduct](https://github.com/nonagenticai/.github/blob/main/CODE_OF_CONDUCT.md).

## Before you start

- **Open an issue before a large change.** It saves you writing code that will not be merged.
- **One concern per pull request.** A feature and a refactor belong in two PRs.
- **Keep the tree green.** If a check fails, say so in the PR rather than working around it.
- **No credentials, ever.** Not in code, tests, fixtures, CI, or commit messages.

## Requirements

The server is Rust end to end. The artifact only *runs* on macOS on Apple Silicon with MLX
and Metal, but the test suite is platform-neutral — the engine boundary is exercised
against a fake SSE peer — so you can build, lint and test on any machine.

- Rust: the toolchain pinned in `rust-toolchain.toml` (installs automatically with rustup).
- To serve a model: macOS on Apple Silicon, Homebrew, and enough unified memory for the
  checkpoint you intend to load. The README explains the memory limits.

## Development setup

```bash
git clone https://github.com/nonagenticai/mlx_inference.git
cd mlx_inference/rust
cargo build --locked
```

## Tests and lint

```bash
cd rust
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Both must pass before a pull request is ready for review. CI runs the same commands on a
GitHub-hosted Linux runner; a local pass and a green CI run are the same gate.

## Branches and commits

Branch names: `feat/`, `fix/`, `docs/`, `chore/`, `ci/` — for example `fix/retry-on-502`.

Commit messages follow `type(scope): summary` in the imperative mood, for example
`fix(sse): close the stream when the engine exits`.

This project uses the [Developer Certificate of Origin](https://developercertificate.org).
Sign off every commit:

```bash
git commit -s -m "fix(sse): close the stream when the engine exits"
```

## Pull requests

- Describe the problem before the solution.
- Link the issue it closes, if there is one.
- Add or update tests for behaviour changes. The contract goldens are executable: changing
  the HTTP surface means changing them deliberately.
- A maintainer reviews every PR. First-time contributors and fork PRs need approval before
  CI runs — that is a GitHub default, not a comment on your patch.

## Reporting a security issue

Do **not** open a public issue. See [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contribution is licensed under Apache-2.0, as described
in [LICENSE](LICENSE).
