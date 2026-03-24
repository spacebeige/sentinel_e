#!/bin/bash
# AWAAZ preflight - run before first start

set -e

echo "=== AWAAZ Preflight Check ==="

# Load .env if exists
if [ -f .env ]; then
    source .env 2>/dev/null || true
fi

# Check required environment variables
check_env() {
    local required=("ASTERISK_HOST" "ASTERISK_ARI_USERNAME" "ASTERISK_ARI_PASSWORD")
    for var in "${required[@]}"; do
        if [ -z "${!var}" ]; then
            echo "  MISSING ENV: $var"
            return 1
        else
            echo "  OK: $var"
        fi
    done
    return 0
}

# Download fastText model
check_fasttext_model() {
    local model_path="${FASTTEXT_MODEL_PATH:-/tmp/lid.176.bin}"
    if [ ! -f "$model_path" ]; then
        echo "[PREFLIGHT] fastText LID model not found. Downloading..."
        wget -q --show-progress \
            https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
            -O "$model_path" || echo "  WARNING: Failed to download fastText model"
        echo "[PREFLIGHT] fastText model downloaded: $model_path"
    else
        echo "[PREFLIGHT] fastText model: OK ($model_path)"
    fi
}

# Check ports
check_ports() {
    echo "[PREFLIGHT] Checking ports..."
    netstat -tlnp 2>/dev/null | grep -q ":8090" \
        && echo "  WARNING: Port 8090 already in use" \
        || echo "  OK: Port 8090 free"
}

# Initialize SQLite database
init_db() {
    echo "[PREFLIGHT] Initialising SQLite schema..."
    python3 -c "
import sqlite3, os
db = sqlite3.connect(os.environ.get('DB_PATH', 'awaaz.db'))
db.executescript('''
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id TEXT PRIMARY KEY,
        session_id TEXT,
        phone_hash TEXT,
        lang TEXT,
        grievance_category TEXT,
        dept_assigned TEXT,
        priority TEXT DEFAULT 'NORMAL',
        complaint_summary TEXT,
        state TEXT DEFAULT 'NEW',
        created_at TEXT,
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS citizens (
        phone_hash TEXT PRIMARY KEY,
        lang TEXT,
        accent_region TEXT,
        last_call TEXT,
        total_complaints INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS call_history (
        session_id TEXT PRIMARY KEY,
        lang TEXT,
        turns INTEGER,
        state TEXT,
        is_emergency INTEGER,
        duration_s REAL,
        ticket_id TEXT,
        created_at TEXT
    );
''')
db.commit()
print('  OK: SQLite schema ready')
"
}

echo ""
echo "[PREFLIGHT] Checking environment..."
check_env && echo "  ✓ All required env vars set" || echo "  ✗ Missing env vars"

check_fasttext_model
check_ports
init_db

echo ""
echo "=== Preflight complete. Run: docker compose up -d ==="
