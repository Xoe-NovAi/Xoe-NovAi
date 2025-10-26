# Xoe-NovAi Guide Batch Writer — Session Log

## Purpose
This file tracks the progress, findings, and corrections across all Group guide-writing 
sessions. It serves as the "tribal knowledge" bridge, ensuring continuity and preventing 
regression of prior work.

## Session Index

### Group 1 — Foundation & Architecture [2025-10-20]
- **Artifacts**: xnai-group1-artifact1-foundation.md (Sections 0-1, 4 patterns)
- **Patches verified**: N/A (Group 1, baseline)
- **Prior corrections**: N/A
- **Issues filed**: 
  - "Verify FAISS index path consistency across Dockerfiles"
- **Blockers**: None
- **Next group note**: Group 2 to verify 197 env vars and implement validate_config(strict=False)

### Group 2 — Prerequisites & Configuration [2025-10-21]
- **Artifacts**: xnai-group2-artifact1-prereqs.md, xnai-group2-artifact2-config.md 
  (Sections 2, 4, 5; 197 vars, 23 config.toml sections)
- **Patches verified**: N/A (Group 2, pre-ChatGPT review)
- **Prior corrections applied**: 
  - ✅ FAISS path checked: consistent in all files
- **Issues filed**:
  - "Add --non-strict flag to validate_config.py"
  - "Vectorstore: add allow_dangerous_deserialization = false default"
- **Blockers**: None
- **Next group note**: Group 3 to verify lazy config() patches and health checks

### Group 3 — Docker & Health Checks [2025-10-22]
- **Artifacts**: xnai-group3-artifact4-docker.md, xnai-group3-artifact5-health.md 
  (Sections 6, 7, 13.1-13.3; 7 health targets, multi-stage builds)
- **Patches verified**: 
  - ⚠️ Lazy `get_config()` patches NOT YET APPLIED to Dockerfile.api (ChatGPT patch pending)
  - ⚠️ WARNING log for dangerous deserialization NOT IMPLEMENTED
- **Prior corrections applied**:
  - ✅ FAISS path confirmed consistent
  - ⚠️ validate_config(strict=False) — patch exists but not applied
- **Issues filed**:
  - "Apply lazy config() patches to all Dockerfiles"
  - "Add WARNING log when FAISS deserialization enabled"
- **Blockers**: 
  - ChatGPT patches (config_loader, dependencies, chainlit) must be applied before 
    health checks run in production
- **Next group note**: Group 4 to verify datetime serialization and URL security patches

### Group 4 — FastAPI, Chainlit, CrawlModule [2025-10-22]
- **Artifacts**: xnai-group4-artifact6-fastapi.md, xnai-group4-artifact7-chainlit.md, 
  xnai-group4-artifact8-crawlmodule.md (Sections 8-11; Pattern 2&3, URL validation, tests)
- **Patches verified**:
  - ⚠️ Chainlit datetime serialization — ISO `start_time_iso` patch provided but 
    verification needed
  - ⚠️ Domain extension attack test case — in patch but verify test_crawl_allowlist.py 
    includes it
- **Prior corrections applied**:
  - ⚠️ Lazy config() and other ChatGPT patches still pending application
  - ✅ URL validation tests comprehensive (except noted domain extension edge case)
- **Issues filed**:
  - "Verify Chainlit datetime serialization patch applied everywhere"
  - "Add domain extension attack test case to test_crawl_allowlist.py"
- **Blockers**: 
  - ChatGPT patch bundle (3 code files + CI workflow + tests) must be applied and 
    committed before Group 5 proceeds with operational sections
- **Next group note**: **GROUP 5 CRITICAL**: Verify all ChatGPT patches applied before 
  starting Section 11 (ingestion/checkpointing) and Section 13 (monitoring/troubleshooting)

### Group 5 — Operations & Quality [STARTING]
- **Artifacts**: TBD (Sections 11, 13.4-13.6; checkpointing, metrics, troubleshooting)
- **Patches verified**: [TO BE FILLED BY GROUP 5]
- **Prior corrections to check**:
  - [ ] Group 2: validate_config(strict=False) flag added
  - [ ] Group 2: config.toml [vectorstore] has allow_dangerous_deserialization = false
  - [ ] Group 3: lazy get_config() applied to Dockerfile.api and dependencies.py
  - [ ] Group 3: WARNING log added for FAISS dangerous deserialization
  - [ ] Group 4: Chainlit ISO start_time_iso patch applied everywhere
  - [ ] Group 4: test_crawl_allowlist.py includes domain extension attack case
- **Issues filed**: [TO BE FILLED]
- **Blockers**: [TO BE FILLED]
- **Next group note**: [TO BE FILLED]

---

## Patch Application Tracker

| Patch                                          | File(s)            | Status    | Applied By | Date |
| ---------------------------------------------- | ------------------ | --------- | ---------- | ---- |
| config_loader: `validate_config(strict=False)` | config_loader.py   | ⚠️ Pending | —          | —    |
| dependencies: lazy `get_config()`              | dependencies.py    | ⚠️ Pending | —          | —    |
| chainlit_app: ISO datetime serialization       | chainlit_app.py    | ⚠️ Pending | —          | —    |
| scripts: test_deployment.sh                    | scripts/           | ⚠️ Pending | —          | —    |
| tests: config_loader, crawl_allowlist          | tests/             | ⚠️ Pending | —          | —    |
| CI: .github/workflows/ci_integration.yml       | .github/workflows/ | ⚠️ Pending | —          | —    |

---

## Critical Path Checklist (Group 5 Entry Gate)

Before Group 5 generates Section 11 & 13 artifacts, the following must be verified ✅:

- [ ] ChatGPT patch bundle applied and committed (all 3 code files + tests + CI workflow)
- [ ] FAISS index path consistency verified across all files
- [ ] Prior group issues resolved or filed as GitHub issues
- [ ] Dockerfile.api uses lazy config() and logs WARNING for dangerous deserialization
- [ ] Chainlit uses ISO start_time_iso for datetime storage
- [ ] test_crawl_allowlist.py includes domain extension attack test case
- [ ] docs/GROUP_SESSIONS_LOG.md updated with Group 5 entry gate status

---

## Notes for Future Groups

- **Patch velocity**: Apply patches within 24 hours of generation to avoid drift between 
  design and implementation.
- **Test coverage**: Each patch should include unit tests; verify tests pass before marking 
  patch as "applied."
- **Cross-artifact consistency**: Verify FAISS paths, config keys, and logging standards 
  remain consistent across all artifacts.