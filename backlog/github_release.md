## GitHub release checklist

1. **License**
   - [x] Decision: MIT.
   - [x] Add LICENSE file with chosen text.
   - [x] Update README header with license badge/info.

2. **Repository visibility**
   - [ ] Public or private confirmed.
   - [ ] Access list (if private) documented.

3. **CI pipeline**
   - [x] Python version(s) for CI: 3.10.
   - [x] Steps: `python3 -m compileall consolidator`, `python scripts/smoke_accuracy.py`.
   - [ ] Optional: lint/format actions (пока не требуются).
   - [x] Workflow file добавлен: `.github/workflows/ci.yml`.

4. **Documentation links**
   - [x] Ensure README references `docs/testing.md` for real-world runs & control hash.
   - [x] Include commands for control hash (GNU/macOS variants).
   - [x] Mention limitations (1–100 000 sources, non-nested directories, etc.).

5. **Smoke/test artifacts**
   - [x] Record latest smoke run outputs (summary table, log behavior).
   - [ ] Attach screenshot or description if TUI is part of release.

6. **Release notes**
   - [ ] Summaries of key features (Unicode names, retries, longest-name rule).
   - [ ] Mention `consolidator_errors.log` examples and how to interpret them.

Use this checklist once license/visibility/CI decisions are confirmed by Eugene.
