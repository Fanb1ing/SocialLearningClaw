# Repository Working Agreement

## Durable task memory

After completing any user task in this repository, update
`docs/project_memory.md` with the date, outcome, important design decisions,
verification performed, and any unfinished follow-up. Keep it concise and do
not record API keys, credentials, private data, or generated run artifacts.

At the start of a new session, read `docs/project_memory.md` before changing
the project. Verify important claims against the current source because the
memory is a handoff aid, not a replacement for code inspection.

## Project conventions

- Use `.venv/bin/python` for local checks.
- Preserve user changes and untracked input files such as `temp.md`.
- The new layered schema is grounded in `socialclaw.memory`; a schema node
  must cite durable memory IDs as evidence.
- The legacy `Concept`/`Relation` graph remains available for existing ARC
  runners until migration is explicitly requested.
