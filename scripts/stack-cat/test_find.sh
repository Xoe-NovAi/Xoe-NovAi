#!/usr/bin/env bash
# Direct test of find command
ROOT_DIR="/home/arcana-novai/Documents/XNAi-v0_1_3-clean"

echo "Test 1: Direct find for Dockerfile*"
find "$ROOT_DIR" -maxdepth 1 -type f -name "Dockerfile*"

echo ""
echo "Test 2: With stderr suppression"
find "$ROOT_DIR" -maxdepth 1 -type f -name "Dockerfile*" 2>/dev/null

echo ""
echo "Test 3: Check what's actually in ROOT_DIR"
ls -la "$ROOT_DIR" | grep -i docker

echo ""
echo "Test 4: Pattern matching with variable"
pattern="Dockerfile*"
file_part="${pattern##*/}"
echo "Pattern: $pattern"
echo "File part: $file_part"
find "$ROOT_DIR" -maxdepth 1 -type f -name "$file_part"