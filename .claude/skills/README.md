# Vendored skills: superpowers

These skills are a vendored copy of the [superpowers](https://github.com/obra/superpowers)
plugin's `skills/` directory, checked into this repository as **project skills** instead of
being installed as a Claude Code plugin.

| | |
|---|---|
| Upstream | https://github.com/obra/superpowers |
| Version | v6.3.0 |
| Commit | `b36e0829c6d0140e93cfef2ca599b1b07d4a7797` |
| License | MIT — see `LICENSE` (Copyright (c) 2025 Jesse Vincent) |

## Why vendored rather than installed

A plugin install lives in `~/.claude/plugins` on one machine. Checking the skills in here
means they travel with the repository: every clone and every Claude Code session on this
project gets them, with no per-machine setup.

## Changes made to the upstream files

Two deliberate deviations, both forced by the move from plugin to project skills:

1. **Namespace prefixes stripped.** Plugin skills are addressed as `superpowers:<name>`;
   project skills are addressed by bare name. Every cross-reference in the skill bodies was
   rewritten accordingly (`superpowers:brainstorming` → `brainstorming`, and so on).
   Nothing else in the text was touched.
2. **SessionStart hook re-rooted.** Upstream ships `hooks/session-start`, which resolves the
   plugin directory via `${CLAUDE_PLUGIN_ROOT}` and injects the `using-superpowers` skill as
   session context. That variable is only set for installed plugins, so the copy at
   `.claude/hooks/superpowers-session-start` derives its root from the script's own location.
   Only the Claude Code output branch was kept; the upstream Cursor and Copilot CLI branches
   were dropped as irrelevant here.

The rest of the plugin — its marketplace metadata, tests, docs, and the non-Claude agent
directories (`.codex-plugin`, `.cursor-plugin`, `.pi`, …) — was not vendored.

## Updating

Re-copy `skills/` from a newer upstream tag, then re-apply change (1):

```sh
git clone --depth 1 https://github.com/obra/superpowers /tmp/superpowers
cp -a /tmp/superpowers/skills/. .claude/skills/
grep -rl 'superpowers:' .claude/skills/ | xargs sed -i 's/superpowers:\([a-z0-9-]\)/\1/g'
```

Then re-add this README and `LICENSE`, and diff `/tmp/superpowers/hooks/session-start`
against `.claude/hooks/superpowers-session-start` for upstream changes worth porting.

## The skills

| Skill | Use when |
|---|---|
| `brainstorming` | Before any creative work — features, components, behavior changes |
| `dispatching-parallel-agents` | 2+ independent tasks with no shared state |
| `executing-plans` | Executing a written plan in a separate session |
| `finishing-a-development-branch` | Implementation done, deciding how to integrate |
| `receiving-code-review` | Acting on review feedback without performative agreement |
| `requesting-code-review` | Before merging, to verify work meets requirements |
| `subagent-driven-development` | Executing plans with independent tasks in-session |
| `systematic-debugging` | Any bug or test failure, before proposing fixes |
| `test-driven-development` | Any feature or bugfix, before writing implementation |
| `using-git-worktrees` | Feature work needing isolation from the current workspace |
| `using-superpowers` | Conversation start — how to find and use skills |
| `verification-before-completion` | Before claiming work complete, fixed, or passing |
| `writing-plans` | A spec for a multi-step task, before touching code |
| `writing-skills` | Creating, editing, or verifying skills |
