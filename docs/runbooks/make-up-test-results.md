---
status: active
last_updated: 2026-01-09
category: runbook
---

# Make Up Command Test Results

**Date:** 2026-01-09  
**Command:** `make up`  
**Status:** ⚠️ Permission Issue Detected

---

## Test Execution

```bash
$ make up
Starting stack...
cat redis_password.txt | docker secret create redis_password -
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
make: *** [Makefile:102: up] Error 1
```

---

## Issue Analysis

### Problem
The `make up` command fails with a Docker permission error when trying to create a Docker secret.

### Root Cause
1. **Makefile uses `sudo docker compose`** (line 9: `COMPOSE := sudo docker compose`)
2. **Docker secret command** (line 102) is executed without sudo
3. **Permission mismatch:** The secret creation command doesn't have sudo, but Docker daemon requires elevated permissions

### Makefile Code
```makefile
COMPOSE := sudo docker compose

up: ## Start stack
	@echo "Starting stack..."
	cat redis_password.txt | docker secret create redis_password -
	$(COMPOSE) up -d
```

---

## Solutions

### Option 1: Add sudo to secret creation (Quick Fix)
```makefile
up: ## Start stack
	@echo "Starting stack..."
	cat redis_password.txt | sudo docker secret create redis_password -
	$(COMPOSE) up -d
```

### Option 2: Use docker-compose secrets (Recommended)
Instead of Docker secrets, use docker-compose environment variables or volume mounts for the Redis password.

### Option 3: Fix Docker permissions (Long-term)
Add user to docker group:
```bash
sudo usermod -aG docker $USER
# Then logout/login or: newgrp docker
```

---

## Current Workaround

Use docker-compose directly:
```bash
sudo docker compose up -d
```

Or fix the Makefile to use sudo consistently:
```makefile
up: ## Start stack
	@echo "Starting stack..."
	cat redis_password.txt | sudo docker secret create redis_password - || true
	$(COMPOSE) up -d
```

---

## Verification Needed

- [ ] Test with sudo added to secret creation
- [ ] Verify Docker secrets are actually needed (vs. environment variables)
- [ ] Check if docker-compose.yml uses secrets or env vars
- [ ] Test alternative approaches

---

## Fix Applied

**Issue:** Makefile was trying to create Docker secrets, but docker-compose.yml uses environment variables.

**Solution:** Removed Docker secret creation from Makefile. The docker-compose.yml uses `${REDIS_PASSWORD}` from `.env` file, not Docker secrets.

**Updated Makefile:**
```makefile
up: ## Start stack
	@echo "Starting stack..."
	@if [ ! -f .env ]; then \
		echo "Warning: .env file not found. Creating from .env.example..."; \
		cp .env.example .env 2>/dev/null || echo "Error: .env.example not found"; \
	fi
	$(COMPOSE) up -d
```

**Status:** ✅ Fixed (Makefile updated)  
**Note:** Docker secrets are not needed - docker-compose.yml uses environment variables from `.env` file.

**Remaining Issue:** The Makefile uses `sudo docker compose` which requires password in non-interactive environments. This is expected behavior and not a code issue.

**Workaround for Non-Interactive Use:**
```bash
# Option 1: Use docker compose directly (if user is in docker group)
docker compose up -d

# Option 2: Use sudo with password (interactive)
sudo docker compose up -d

# Option 3: Configure passwordless sudo for docker commands (system config)
```

**Recommendation:** Consider removing `sudo` from Makefile if user is in docker group, or document the sudo requirement.

