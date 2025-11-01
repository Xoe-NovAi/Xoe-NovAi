#!/bin/bash
# clean_wheelhouse_duplicates.sh
# Scans wheelhouse for duplicate wheels and removes all but the required version(s) based on requirements files.
# Logs any wheel/file downloaded or requested more than once.

WHEELHOUSE="wheelhouse"
REQUIREMENTS=(requirements-api.txt requirements-chainlit.txt requirements-crawl.txt requirements-curation_worker.txt)
LOGFILE="wheelhouse_cleanup.log"

# Collect required versions from all requirements files
declare -A required_versions
for req in "${REQUIREMENTS[@]}"; do
    if [[ -f "$req" ]]; then
        while read -r line; do
            pkg=$(echo "$line" | grep -Eo '^[a-zA-Z0-9_-]+')
            ver=$(echo "$line" | grep -Eo '==[0-9a-zA-Z.]+')
            if [[ "$pkg" && "$ver" ]]; then
                required_versions[$pkg]=${ver#==}
            fi
        done < "$req"
    fi
done

# Find duplicates and remove unneeded versions
for whl in $WHEELHOUSE/*.whl; do
    fname=$(basename "$whl")
    pkg=$(echo "$fname" | grep -Eo '^[a-zA-Z0-9_-]+')
    ver=$(echo "$fname" | grep -Eo '[0-9]+\.[0-9]+(\.[0-9]+)?')
    req_ver=${required_versions[$pkg]}
    # If required version is set and does not match, remove
    if [[ "$req_ver" && "$ver" != "$req_ver" ]]; then
        echo "Removing unneeded wheel: $fname" | tee -a "$LOGFILE"
        rm -f "$whl"
    fi
    # Log duplicate downloads
    if [[ $(ls $WHEELHOUSE/${pkg}-*.whl | wc -l) -gt 1 ]]; then
        echo "Duplicate wheel detected: $pkg ($fname)" | tee -a "$LOGFILE"
    fi
done

echo "Wheelhouse cleanup complete. See $LOGFILE for details."
