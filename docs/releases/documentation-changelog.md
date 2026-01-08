```markdown
# CHANGES

This file records top-level, human-friendly changes to the documentation and infra strategy.

2026-01-06
- **Critical Security Fixes Implemented:** Resolved 5 major security vulnerabilities
  - Command injection protection: Added input validation for crawl.py and chainlit_app.py /curate command
  - Path traversal protection: Sanitized file paths and IDs to prevent directory traversal attacks
  - Redis security enhancement: Required password validation and protected mode enforcement
  - Async operations framework: Added foundation for converting synchronous operations to async
  - Health check optimization: Implemented caching for expensive LLM/vectorstore checks (5-minute TTL)
  - See docs/runbooks/security-fixes-runbook.md for complete implementation details

2026-01-09
- **Complete documentation reorganization:** Reorganized 110+ files into strategic category structure
  - Created 8 category folders: reference/, howto/, design/, implementation/, runbooks/, releases/, policies/, archive/
  - Reduced top-level files from 79 to 5 (95% reduction)
  - Consolidated archive/ and archived/ into single archive/ with subdirectories
  - Created index files (README.md) for each category folder
  - Moved all duplicates to archive/duplicates/
  - Preserved all historical content in organized archive/
  - Created AI_ASSISTANT_GUIDE.md for AI coding assistant efficiency
  - Updated DOCS_STRATEGY.md with final structure
  - Fixed Makefile `make up` command (removed unnecessary Docker secret creation)
  - See ORGANIZATION_COMPLETE.md and DOCUMENTATION_ORGANIZATION_SUMMARY.md for details

2026-01-04
- Consolidated root docs into `docs/` and created `docs/archive/` for originals.
- Added `docs/UPDATES_RUNNING.md` as the canonical runbook and archived the root snapshot.
- Added `docs/CHANGES.md` and `docs/OWNERS.md` to support doc governance.

```
