---
status: active
last_updated: 2026-01-08
category: howto
---

# GitHub Protocol Guide for Xoe-NovAi

**Complete guide for using GitHub effectively in the Xoe-NovAi project.**

This guide covers best practices for commits, branches, pull requests, issues, and collaboration workflows specific to this project.

---

## 🎯 Quick Start for GitHub Newbies

### Your First Steps
1. **Create GitHub Account:** Sign up at [github.com](https://github.com)
2. **Join the Project:** Get invited to the Xoe-NovAi repository
3. **Set Up Git:** Install Git and configure your identity
4. **Clone the Repository:** Get a local copy of the project

### Essential Setup Commands
```bash
# Configure your identity (do this once)
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"

# Clone the repository
git clone https://github.com/Xoe-NovAi/Xoe-NovAi.git
cd Xoe-NovAi

# Set up the main branch
git checkout main
git pull origin main
```

---

## 📋 GitHub Workflow Overview

### Branching Strategy
```
main (production) ← develop ← feature branches
                          ← bugfix branches
                          ← hotfix branches
```

### Typical Development Flow
1. **Create Feature Branch** from `main`
2. **Make Changes** with clear, atomic commits
3. **Push Branch** to GitHub
4. **Create Pull Request** for review
5. **Address Feedback** and update PR
6. **Merge** when approved

---

## 🔄 Branch Management

### Branch Naming Conventions

#### Feature Branches
```
feature/voice-to-voice-conversation
feature/build-tracking-system
feature/enhanced-makefile
```

#### Bugfix Branches
```
bugfix/chainlit-memory-limit-fix
bugfix/docker-build-performance
```

#### Hotfix Branches (for production issues)
```
hotfix/critical-security-patch
hotfix/database-connection-fix
```

### Creating and Managing Branches

```bash
# Create and switch to a new feature branch
git checkout -b feature/your-feature-name

# Push the branch to GitHub
git push -u origin feature/your-feature-name

# Switch between branches
git checkout main                    # Switch to main
git checkout feature/my-feature      # Switch to feature branch
git checkout -b bugfix/issue-123     # Create and switch to bugfix branch
```

---

## 📝 Commit Best Practices

### Commit Message Format
```
Type: Brief description of change

Optional detailed explanation of what changed and why.

Fixes #123
Related to #456
```

### Commit Types
- **feat:** New feature or enhancement
- **fix:** Bug fix
- **docs:** Documentation changes
- **style:** Code style/formatting changes
- **refactor:** Code refactoring (no functional changes)
- **test:** Test additions/modifications
- **chore:** Maintenance tasks (build, dependencies, etc.)

### Good Commit Examples

```bash
# Feature commit
feat: implement voice-to-voice conversation system

- Add real-time speech-to-text processing
- Integrate Piper ONNX TTS synthesis
- Implement voice activity detection
- Add Chainlit voice controls

Fixes #234

# Bug fix commit
fix: remove memory limit causing Chainlit startup failure

- Remove check_available_memory() function from dependencies.py
- Update error handling for unlimited memory usage
- Add documentation for memory management changes

Closes #345

# Documentation commit
docs: add comprehensive Makefile usage guide

- Document all 27 Makefile targets with examples
- Add voice system deployment instructions
- Include build tracking and wheel management guides
- Provide troubleshooting section

Related to enhancement #456
```

### Commit Guidelines

#### ✅ Do This
- **Atomic Commits:** Each commit should do one thing well
- **Clear Messages:** Explain what and why, not just what
- **Reference Issues:** Use "Fixes #123" or "Closes #456"
- **Test Before Commit:** Ensure code works before committing

#### ❌ Don't Do This
- **Mass Commits:** Don't commit 50 files with "updated code"
- **Vague Messages:** Avoid "fixed bug" or "changes"
- **Mixed Changes:** Don't mix features, fixes, and refactoring
- **Unrelated Changes:** Keep commits focused on single concerns

### Committing Workflow

```bash
# Check what you've changed
git status

# Add specific files or all changes
git add specific-file.py
git add .  # Add all changes (use carefully)

# Commit with a good message
git commit -m "feat: add voice conversation capabilities

- Implement real-time speech recognition
- Add text-to-speech synthesis
- Create voice activity detection

Fixes #789"

# Push your changes
git push origin feature/voice-system
```

---

## 🔄 Pull Request Process

### Creating a Pull Request

1. **Push Your Branch** to GitHub
```bash
git push -u origin feature/your-feature-name
```

2. **Go to GitHub** and navigate to your repository

3. **Click "Compare & Pull Request"** button

4. **Fill Out PR Template:**
   - **Title:** Clear, descriptive title
   - **Description:** What, why, and how
   - **Linked Issues:** Reference related issues
   - **Testing:** How you tested the changes
   - **Screenshots:** If UI changes

### PR Template Example

```markdown
## 🎯 What This PR Does
Implements voice-to-voice conversation system for natural spoken AI interaction.

## 🔧 Technical Changes
- Added real-time speech-to-text processing with Faster Whisper
- Integrated Piper ONNX text-to-speech synthesis
- Implemented voice activity detection and conversation management
- Enhanced Chainlit UI with voice controls

## 🧪 Testing Done
- Voice recognition accuracy testing (95%+ confidence)
- Conversation latency measurement (<500ms end-to-end)
- Error handling validation with graceful fallbacks
- Cross-browser compatibility testing

## 📚 Documentation Updated
- `docs/enhancements/enhancement-voice-to-voice-basic.md`
- `docs/howto/makefile-usage.md`
- `docs/STACK_STATUS.md`

## 🔗 Related Issues
Fixes #234 (Voice Conversation Feature)
Related to #456 (UX Enhancement Epic)

## 🚀 Deployment Notes
- Requires Piper ONNX model download
- No breaking changes to existing APIs
- Backward compatible with text-only mode
```

### PR Review Process

#### For Contributors (You)
1. **Address Feedback:** Make requested changes
2. **Update Commits:** Amend or add commits as needed
3. **Rebase if Needed:** Keep PR up to date with main
4. **Test Thoroughly:** Ensure all changes work

#### For Reviewers
- **Code Quality:** Style, structure, performance
- **Testing:** Adequate test coverage and validation
- **Documentation:** Updated docs and clear comments
- **Breaking Changes:** Proper migration guides if needed

### Updating Your PR

```bash
# Make changes based on review feedback
git add .
git commit -m "Address review feedback: improve error handling"

# Or amend the last commit
git commit --amend -m "feat: implement voice system (updated)"

# Push the updates
git push origin feature/your-feature-name
```

---

## 🐛 Issue Management

### Issue Types

#### 🐛 Bug Reports
```markdown
**Bug Description**
Clear description of what's wrong

**Steps to Reproduce**
1. Go to voice chat interface
2. Click "Start Voice Chat"
3. Speak into microphone
4. See error: "Voice processing failed"

**Expected Behavior**
Voice should be recognized and processed

**Environment**
- OS: Ubuntu 22.04
- Browser: Chrome 120
- Device: Ryzen 7 Desktop

**Additional Context**
Error occurs consistently, works in text mode
```

#### 🚀 Feature Requests
```markdown
**Feature Summary**
Voice-to-voice conversation system

**Problem Statement**
Users must type to interact with AI, creating barriers for accessibility

**Proposed Solution**
Implement natural voice conversation with speech recognition and synthesis

**Alternative Solutions**
- Voice commands only (limited)
- Third-party voice APIs (expensive)

**Additional Context**
Would significantly improve UX for disabled users and hands-free operation
```

#### 📚 Documentation Issues
```markdown
**Documentation Problem**
Makefile usage guide is missing voice system commands

**Current State**
Only covers basic Docker commands

**Suggested Fix**
Add section for voice testing and deployment:
- make voice-test
- make voice-build
- make voice-up

**Location**
docs/howto/makefile-usage.md
```

### Issue Labels
- `bug` - Something isn't working
- `enhancement` - New feature request
- `documentation` - Docs need updating
- `good first issue` - Beginner-friendly
- `help wanted` - Community contribution welcome
- `priority: high` - Urgent fix needed

---

## 🔄 Keeping Your Branch Updated

### Rebasing on Main

```bash
# Fetch latest changes from remote
git fetch origin

# Rebase your feature branch on latest main
git checkout feature/your-feature
git rebase origin/main

# If there are conflicts, resolve them, then:
git add resolved-files
git rebase --continue

# Force push the updated branch
git push origin feature/your-feature --force-with-lease
```

### Squashing Commits (Optional)

```bash
# Interactive rebase to squash commits
git rebase -i HEAD~3  # Last 3 commits

# In the editor, change 'pick' to 'squash' for commits to combine
# Edit the commit message

# Force push the squashed commits
git push origin feature/your-feature --force-with-lease
```

---

## 🛡️ Code Quality Standards

### Pre-Commit Checks

#### Run Tests
```bash
make test                    # Run all tests
pytest tests/                # Run specific tests
```

#### Code Quality
```bash
# Check for linting issues
flake8 app/ scripts/

# Format code
black app/ scripts/

# Type checking (if configured)
mypy app/
```

#### Build Validation
```bash
make build-analyze          # Analyze build state
make check-duplicates       # Check for duplicate packages
make voice-test             # Test voice system
```

### CI/CD Integration

The project uses GitHub Actions for automated testing:

- **Pull Request Checks:** Tests run automatically on PR creation
- **Code Quality:** Linting and formatting validation
- **Build Testing:** Docker build verification
- **Voice System Testing:** Automated voice feature validation

---

## 📚 Documentation Standards

### File Organization
```
docs/
├── README.md              # Main navigation
├── START_HERE.md          # Quick start guide
├── STACK_STATUS.md        # Current system status
│
├── howto/                 # Step-by-step guides
│   ├── quick-start.md
│   ├── docker-setup.md
│   └── makefile-usage.md  # ← Your new guide goes here
│
├── enhancements/          # Feature documentation
├── implementation/        # Technical implementation
├── runbooks/             # Operational procedures
└── policies/             # Project policies
```

### Documentation Frontmatter
```yaml
---
status: active
last_updated: 2026-01-08
category: howto
---
```

### Writing Guidelines
- **Clear Headings:** Use descriptive, actionable headings
- **Code Examples:** Provide working code snippets
- **Cross-References:** Link to related documentation
- **Step-by-Step:** Break down complex processes
- **Troubleshooting:** Include common issues and solutions

---

## 🚨 Emergency Procedures

### If You Accidentally Commit to Main
```bash
# Don't panic! Create a revert commit
git revert HEAD
git push origin main
```

### If You Lose Local Changes
```bash
# Check reflog for lost commits
git reflog

# Restore from reflog
git checkout <commit-hash>
```

### If Merge Conflicts Block Everything
```bash
# Abort the merge/rebase
git merge --abort
git rebase --abort

# Get help from team member
# Create new branch from clean state
git checkout -b feature/fresh-start origin/main
```

---

## 🎯 Best Practices Summary

### Daily Workflow
1. **Start Day:** `git pull origin main` to get latest changes
2. **Work:** Create feature branches for all changes
3. **Commit Often:** Small, focused commits with clear messages
4. **Push Regularly:** Keep your branches up to date
5. **Test Thoroughly:** Use `make test` and `make voice-test`

### Communication
- **Issues for Bugs/Features:** Use GitHub Issues for tracking
- **PRs for Code Changes:** All code goes through pull requests
- **Discussions for Ideas:** Use GitHub Discussions for brainstorming
- **Wiki for Knowledge:** Update documentation as you learn

### Quality Assurance
- **Test Your Code:** Always run tests before pushing
- **Review Your PR:** Read your own PR description carefully
- **Address Feedback:** Be responsive to reviewer comments
- **Document Changes:** Update docs for user-facing changes

---

## 📞 Getting Help

### Resources
- **GitHub Docs:** [github.com/docs](https://docs.github.com)
- **Git Cheat Sheet:** Search for "git cheat sheet"
- **Project Wiki:** Check repository wiki for project-specific info
- **Team Chat:** Ask questions in project communication channels

### When to Ask for Help
- **Stuck for >30 minutes** on a Git/GitHub issue
- **Merge conflicts** you can't resolve
- **Breaking changes** that affect other team members
- **Security concerns** or sensitive operations

### Escalation Path
1. **Check Documentation:** Project docs and GitHub guides
2. **Search Issues:** See if others had similar problems
3. **Ask Team Member:** Direct message for quick questions
4. **Create Issue:** For complex problems needing discussion

---

## 🎉 Success Metrics

### You're Doing Great If:
- ✅ **Consistent Commits:** Clear, descriptive commit messages
- ✅ **Active PRs:** Regular pull request creation and updates
- ✅ **Issue Management:** Proper bug reports and feature requests
- ✅ **Code Reviews:** Constructive feedback on others' code
- ✅ **Documentation:** Keeping docs updated with your changes

### Advanced GitHub User Checklist:
- [ ] Created your first pull request
- [ ] Successfully merged a feature branch
- [ ] Resolved merge conflicts independently
- [ ] Reviewed someone else's pull request
- [ ] Updated project documentation
- [ ] Used GitHub Issues effectively
- [ ] Participated in project discussions

---

## 🚀 Next Steps

1. **Set Up Your Environment:** Follow the setup commands above
2. **Read the Contributing Guide:** Check project CONTRIBUTING.md if it exists
3. **Start Small:** Fix a small bug or add a minor feature
4. **Ask Questions:** Don't hesitate to ask for help
5. **Learn Continuously:** GitHub skills improve with practice

---

**Remember:** Everyone starts as a beginner. The important thing is to communicate clearly, follow the established patterns, and be willing to learn. Welcome to the Xoe-NovAi team! 🚀

---

**Last Updated:** 2026-01-08  
**For:** Xoe-NovAi Contributors (All Skill Levels)  
**Maintained By:** Xoe-NovAi Development Team