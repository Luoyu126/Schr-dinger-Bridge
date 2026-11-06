# AGENTS.md

Guidelines for any AI coding agent working in this repository.

## Repository

- Remote: `origin` → https://github.com/Luoyu126/Schr-dinger-Bridge.git
- Default branch on the remote: `main`

## Git workflow

### Commit and push promptly
- Commit and push to the remote (`git push origin <branch>`) as soon as a unit of work is finished. Don't let finished work pile up locally.
- Before you push, make sure the code at least imports/runs and any existing tests pass.

### Each commit is one self-contained module
- Every commit has to stand on its own (自成一体): one complete feature, module, or piece of functionality.
- Split work by functional module. **Do not** commit all code and docs in one large commit.
- Example split for a new feature:
  1. Core module or algorithm implementation
  2. Its config, CLI, or entry-point wiring
  3. Its tests
  4. Its documentation or README updates
- Commit messages are short and descriptive, in imperative mood (e.g. `Add Sinkhorn-based SB solver`, `Add training script for 2D toy datasets`).

### No AI attribution — ever
- **Never** list yourself (the AI agent/assistant) as an author, co-author, co-advisor, contributor, or anything similar.
- No `Co-Authored-By:` trailers, no "Generated with ..." lines, no AI tool names or links in commit messages, PR titles or descriptions, code comments, docs, or file headers.
- Commits are made only under the repository owner's configured git identity. Don't change `user.name` / `user.email` to anything AI-related.
- Principle: the agent must not be visible anywhere in the repository history or contents.

## Filesystem access

- You may read files anywhere on the machine, but only write, create, or delete files under `/usr0/home/shengyuf/yunyi/`. Never modify anything outside it.
- Writes include temporary files, caches, cloned reference repos, logs, checkpoints, and experiment outputs. Put them under `/usr0/home/shengyuf/yunyi/` (for example a git-ignored directory in this repo), not in `/tmp` or elsewhere in the home directory.

## Python environments

- You may create your own environment, but only under `/usr0/home/shengyuf/yunyi/` (e.g. `/usr0/home/shengyuf/yunyi/envs/<name>`).
- Never modify, install into, update, or delete any existing environment (conda `base`, other conda envs, system Python). Running an existing interpreter read-only is fine; installing packages into it is not.
- Keep package caches inside `/usr0/home/shengyuf/yunyi/` too (e.g. set `PIP_CACHE_DIR`), and avoid tools that write registries outside it (e.g. `conda create` updates `~/.conda/environments.txt`; prefer `python -m venv`).

## General

- Don't commit secrets, large datasets, checkpoints, or generated outputs. Add them to `.gitignore`.
- Keep changes focused. Don't reformat or touch unrelated files in the same commit.
