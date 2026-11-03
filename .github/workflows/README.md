# workflows: testing and releasing dovata

Two GitHub Actions workflows test every change and publish each new version. They
follow the jadelang repo's workflows, `joericks1998/jade`, step for step, so both
projects release the same way.

## How a release happens

A release starts with a version bump and nothing else:

1. Change `version` in the root `Cargo.toml`, for example to `3.0.2`, and run
   `cargo build` so `Cargo.lock` follows. CI builds with `--locked`, so a bump without
   the lockfile fails.
2. Merge the change to `main`.
3. `ci.yml` builds and tests `main`. When that passes, its `tag-release` job reads the
   version, and if the tag `v3.0.2` does not exist yet, it pushes the tag.
4. A tag pushed with the workflow's own token starts no other workflow, so
   `tag-release` sends a `repository_dispatch` event named `release` instead.
5. `release.yml` builds both programs on each platform, checks that
   `dovata --version` matches the tag, and publishes a GitHub release with the
   archives, `checksums.txt` and generated release notes.

A merge that does not change the version tags nothing and releases nothing.

Pushing a `v*` tag by hand also starts `release.yml`. So does a manual run with a tag
as input, such as `gh workflow run release.yml -f tag=v3.0.1`, which attaches the
programs to that tag's release even if the release already exists.

## What a release holds

| file | contents |
|---|---|
| `dovata-macos-arm64.tar.gz` | `dovata` and `gguf-inspect` for Apple Silicon |
| `dovata-linux-x86_64.tar.gz` | the same for x86_64 Linux, built against glibc |
| `checksums.txt` | the SHA-256 of each archive |

`refback` is not in the archives. It is the test-only reference backend and is never
published. The root `install.sh` downloads the archive for the machine it runs on,
checks it against `checksums.txt`, and puts both programs in `/usr/local/bin`.

## Files

- `ci.yml` runs on every pull request and push to `main`, on Ubuntu and macOS. It
  runs the workspace build and tests, the tests with every capability compiled out, and
  clippy. On macOS it also runs the quantize tests through Homebrew's llama.cpp and
  refback's Metal tests. It ends with `dovata machine info` on the runner. On `main` it
  also runs `tag-release`.
- `release.yml` builds and publishes a release, as above.

## Where it differs from jadelang

- There is no `cargo fmt --check`. dovata's function signatures put each parameter on
  its own line with the brace below, which stable rustfmt cannot produce.
- There is no LLVM step, since dovata does not link LLVM.
- A manual run of `release.yml` builds the tag it is given. jadelang's manual run
  builds the branch it was started from and publishes it under the tag.
- The release build checks the programs' version against the tag.

## Who uses it

The root `install.sh` and the README's install line download what `release.yml`
publishes. The Jade compiler takes the library from a release tag as a cargo
dependency, so the tags `tag-release` pushes are its versions too.
