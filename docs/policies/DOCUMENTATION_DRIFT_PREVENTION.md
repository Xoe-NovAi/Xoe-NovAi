---
status: active
last_updated: 2026-01-09
category: policy
---

# Documentation Drift Prevention Policy

**Purpose:** Prevent documentation from becoming outdated and ensure AI assistants have accurate, current information.

---

## Principles

### 1. Single Source of Truth
- **STACK_STATUS.md** is the canonical reference for current stack state
- All documentation should reference STACK_STATUS.md for current implementations
- When stack changes, update STACK_STATUS.md first

### 2. Version Alignment
- All documentation must include version information
- Version strings must match actual codebase versions
- Use semantic versioning consistently

### 3. Automated Validation
- Documentation should be validated against codebase
- Use scripts to check for outdated references
- CI/CD should flag documentation inconsistencies

### 4. Regular Reviews
- Weekly review of STACK_STATUS.md
- Monthly review of all documentation
- Quarterly comprehensive audit

---

## Implementation Strategy

### Current Stack Tracking

**File:** `docs/STACK_STATUS.md`
- **Purpose:** Single source of truth for current stack
- **Update Frequency:** Immediately when stack changes
- **Format:** Markdown with frontmatter
- **Auto-update Flag:** `auto_update: true`

### Documentation Structure

```
docs/
├── STACK_STATUS.md          # Current stack (single source of truth)
├── reference/               # Technical references (reference STACK_STATUS.md)
├── howto/                   # Guides (reference STACK_STATUS.md)
├── implementation/          # Implementation guides (reference STACK_STATUS.md)
├── design/                  # Design decisions (reference STACK_STATUS.md)
└── releases/                # Historical releases (versioned)
```

### Update Workflow

1. **Stack Change Made:**
   - Update codebase
   - Update `STACK_STATUS.md` immediately
   - Update relevant documentation files
   - Update version numbers

2. **Documentation Created:**
   - Reference `STACK_STATUS.md` for current state
   - Include version information
   - Link to STACK_STATUS.md

3. **Regular Review:**
   - Check STACK_STATUS.md against codebase
   - Verify all documentation references are current
   - Update outdated information

---

## AI Assistant Guidelines

### For AI Coding Assistants

1. **Always Check STACK_STATUS.md First**
   - Before making assumptions about stack
   - Before suggesting implementations
   - Before updating documentation

2. **Reference STACK_STATUS.md in Responses**
   - When discussing current stack
   - When suggesting changes
   - When creating new documentation

3. **Update STACK_STATUS.md When Stack Changes**
   - Immediately after code changes
   - Before updating other documentation
   - Include rationale for changes

### For Human Developers

1. **Update STACK_STATUS.md First**
   - When adding new components
   - When changing implementations
   - When making technology decisions

2. **Reference STACK_STATUS.md in Documentation**
   - Link to STACK_STATUS.md
   - Don't duplicate information
   - Keep documentation focused

3. **Review Regularly**
   - Weekly: Check STACK_STATUS.md accuracy
   - Monthly: Review all documentation
   - Quarterly: Comprehensive audit

---

## Validation Checks

### Automated Checks (Recommended)

1. **Version Consistency**
   - Check version strings match across files
   - Verify STACK_STATUS.md version matches codebase
   - Flag mismatches

2. **Reference Validation**
   - Check all documentation references STACK_STATUS.md
   - Verify links are valid
   - Flag broken references

3. **Code-Documentation Alignment**
   - Check code matches documentation
   - Verify implementations match descriptions
   - Flag discrepancies

### Manual Checks

1. **Weekly Review**
   - Review STACK_STATUS.md
   - Check for outdated information
   - Update as needed

2. **Monthly Review**
   - Review all documentation
   - Check for drift
   - Update outdated content

3. **Quarterly Audit**
   - Comprehensive review
   - Update all documentation
   - Archive outdated content

---

## Best Practices

### Documentation Creation

1. **Always Reference STACK_STATUS.md**
   ```markdown
   See [STACK_STATUS.md](../STACK_STATUS.md) for current stack implementation.
   ```

2. **Include Version Information**
   ```markdown
   **Version:** v0.1.4-stable (as of 2026-01-09)
   **Reference:** [STACK_STATUS.md](../STACK_STATUS.md)
   ```

3. **Link to Current Implementation**
   ```markdown
   Current implementation: [STACK_STATUS.md#tts](../STACK_STATUS.md#text-to-speech-tts)
   ```

### Documentation Updates

1. **Update STACK_STATUS.md First**
   - Make stack change
   - Update STACK_STATUS.md immediately
   - Then update other documentation

2. **Version All Changes**
   - Include version in frontmatter
   - Update last_updated date
   - Note change rationale

3. **Cross-Reference Updates**
   - Update all references to changed component
   - Verify links still work
   - Update related documentation

---

## Tools & Automation

### Recommended Tools

1. **Documentation Validator**
   - Check version consistency
   - Validate references
   - Flag outdated information

2. **Link Checker**
   - Verify all links work
   - Check for broken references
   - Validate cross-references

3. **Code-Documentation Sync**
   - Compare code to documentation
   - Flag mismatches
   - Suggest updates

### CI/CD Integration

1. **Pre-commit Hooks**
   - Check STACK_STATUS.md updated
   - Validate version consistency
   - Verify references

2. **Automated Tests**
   - Test documentation links
   - Validate version strings
   - Check for drift

3. **Regular Audits**
   - Weekly automated checks
   - Monthly comprehensive review
   - Quarterly full audit

---

## Success Metrics

### Drift Prevention

- **Target:** <5% documentation drift
- **Measurement:** Automated checks + manual review
- **Frequency:** Weekly

### Update Timeliness

- **Target:** STACK_STATUS.md updated within 24 hours of stack change
- **Measurement:** Change log review
- **Frequency:** Daily

### Reference Accuracy

- **Target:** 100% valid references
- **Measurement:** Link checker
- **Frequency:** Weekly

---

## Review Schedule

- **Daily:** Check for stack changes
- **Weekly:** Review STACK_STATUS.md
- **Monthly:** Review all documentation
- **Quarterly:** Comprehensive audit

---

**Last Updated:** 2026-01-09  
**Next Review:** 2026-01-16 (weekly)  
**Policy Owner:** Documentation Team

