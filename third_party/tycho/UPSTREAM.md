# Tycho upstream snapshot

- Project: `NIMI-research/Tycho`
- Repository: <https://github.com/NIMI-research/Tycho>
- Upstream commit: `f68912a764372ead0a610db2e1c011d41ce5197e`
- Imported: 2026-08-31
- License: Apache-2.0; see `LICENSE` in this directory.
- Source package location in this repository: `tycho/`
- Upstream parity tests: `third_party/tycho/tests/`

The copied upstream tests remain byte-for-byte identical to the pinned
revision. SocialLearningClaw extensions primarily live under `socialclaw/v3/`;
the narrow integration patches below are intentionally isolated.

The upstream `CITATION.cff` and `PUBLIC_RELEASE_MANIFEST.json` are retained in
this directory for attribution and release-integrity review. Any future
upstream upgrade must update the commit above, replace the snapshot and tests,
run the upstream parity suite, and document source differences here.

`third_party/tycho/tycho` is a relative symlink to the canonical top-level
snapshot. It preserves the upstream tests' repository-relative source lookups
without maintaining a second source copy.

## Local packaging differences

The upstream distribution metadata is not nested into this repository. The
root `pyproject.toml` packages the unmodified top-level `tycho` package and
declares the runtime dependencies needed by both projects. The authoritative
upstream dependency pins remain recorded in the reviewed upstream commit and
the V3 development documentation.

## Patch inventory

1. `tycho/agent/agent.py` resolves its workspace and tool executor through class
   attributes that default to the original `GameWorkspace` and `ToolExecutor`;
   upstream behavior is unchanged, while the EFPS approach can supply narrow
   subclasses without copying `reset()`.
2. `tycho/harness/_run_extension.py` registers the `tycho_efps` approach and
   includes `socialclaw/v3` in Tycho's immutable policy-source fingerprint; it
   also exposes hashes for the local game implementations.
3. `tycho/harness/harness.py` accepts `TYCHO_ENVIRONMENTS_DIR`, falling back to
   the original root `environment_files/`; the V3 wrapper points workers at the
   repository's existing pinned `third_party/arc_agi3_games/` inventory.
4. `tycho/harness/run_parallel.py` merges extension-provided execution-source
   identities into both coordinator and worker run specs, so resume fails closed
   if a local game source or metadata fingerprint changes.
5. `tycho/harness/harness.py`, `run_parallel.py`, and `run_spec.py` expose
   explicit per-level action and level-count limits for bounded development
   experiments. The limits propagate into worker subprocesses and are part of
   the immutable run identity, so a resumed run cannot silently change them.
6. `tycho/workspace/sandbox.py` adds a Linux Bubblewrap fallback. It runs the
   workspace with a read-only allowlisted filesystem, a private network
   namespace, cleared environment, dropped capabilities, and only the current
   game workspace writable. This preserves isolation on development hosts where
   Docker is installed but its daemon socket is unavailable to the user.
