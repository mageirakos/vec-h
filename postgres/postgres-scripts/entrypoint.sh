#!/usr/bin/env bash
set -euo pipefail

# Postgres lifecycle manager for standalone container usage (SBATCH / docker-compose).
#
# Handles two runtime modes:
#   - Root (docker/docker-compose): uses su - postgres for pg operations
#   - Non-root (podman/enroot on HPC): runs pg operations directly as current user
#
# Env vars:
#   PGDATA              - data directory (must be mounted for persistence)
#   PG_TUNING_CONF      - path to .conf file to append to postgresql.conf (optional)
#   POSTGRES_USER       - superuser name (default: postgres)
#   POSTGRES_PASSWORD   - superuser password (default: 1234)
#   POSTGRES_DB         - database to create (default: vech)

# The upstream pgvector/postgres image installs binaries at
# /usr/lib/postgresql/<major>/bin/ and relies on a Dockerfile ENV PATH directive
# to expose them. Some enroot/pyxis launch paths (Alps daint) drop that
# augmentation, so re-add it here explicitly. Glob across major versions so
# this keeps working when the base image bumps from pg17 → pg18 etc.
for _pgbin in /usr/lib/postgresql/*/bin; do
    if [ -d "$_pgbin" ] && [ -x "$_pgbin/initdb" ]; then
        export PATH="$_pgbin:$PATH"
        break
    fi
done
unset _pgbin

# Tell client tools (psql, pg_isready) to look for the unix socket in /tmp,
# matching unix_socket_directories='/tmp' that we write to postgresql.conf
# during init below. Default ('/var/run/postgresql') doesn't exist inside the
# enroot container.
export PGHOST=/tmp

: "${PGDATA:=/var/lib/postgresql/data}"
: "${POSTGRES_USER:=postgres}"
: "${POSTGRES_PASSWORD:=1234}"
: "${POSTGRES_DB:=vech}"

export PGDATA

# --- Detect runtime mode ---
# Root: docker-compose (rootful or rootless — root inside container either way)
# Non-root: enroot/pyxis on HPC (container runs as SLURM user's UID)
if [ "$(id -u)" -eq 0 ]; then
    RUN_AS="su - postgres -c"
    echo "Running as root — will su to postgres for pg operations"
else
    # Non-root: run pg commands directly as current user.
    # Postgres doesn't care about username, only that the UID owns PGDATA.
    RUN_AS="bash -c"
    echo "Running as UID $(id -u) — pg operations run directly (HPC/podman mode)"
fi

# Helper: run a command as the postgres-owning user
pg_run() {
    $RUN_AS "$1"
}

# --- Cleanup on exit ---
cleanup() {
    echo "Stopping postgres..."
    # Force a CHECKPOINT first. The subsequent `pg_ctl stop -m fast` does its
    # own final checkpoint, but its wait is capped at 60s by default — too
    # short to flush e.g. a fresh 13 GB data load on Lustre. CHECKPOINT here
    # runs synchronously with no client-side timeout, so we wait as long as
    # postgres needs. Once it returns, pg_ctl stop's checkpoint is near-empty
    # and shutdown completes within seconds.
    pg_run "psql -U $POSTGRES_USER -d postgres -c 'CHECKPOINT;'" 2>/dev/null || true
    pg_run "pg_ctl stop -D '$PGDATA' -m fast" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

# --- Ensure PGDATA directory permissions ---
if [ "$(id -u)" -eq 0 ]; then
    chown -R postgres:postgres "$PGDATA" 2>/dev/null || true
fi

# --- Init or start ---
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "=== Initializing new database cluster ==="
    pg_run "initdb -D '$PGDATA' --username='$POSTGRES_USER' --pwfile=<(echo '$POSTGRES_PASSWORD')"

    # Apply tuning config
    if [ -n "${PG_TUNING_CONF:-}" ] && [ -f "$PG_TUNING_CONF" ]; then
        echo "Applying tuning config: $PG_TUNING_CONF"
        echo "" >> "$PGDATA/postgresql.conf"
        echo "# --- pgtuning (applied by entrypoint.sh) ---" >> "$PGDATA/postgresql.conf"
        cat "$PG_TUNING_CONF" >> "$PGDATA/postgresql.conf"
    fi

    # Allow local connections without password (inside container only)
    echo "host all all 0.0.0.0/0 md5" >> "$PGDATA/pg_hba.conf"
    echo "local all all trust" >> "$PGDATA/pg_hba.conf"
    # Listen on all interfaces
    echo "listen_addresses = '*'" >> "$PGDATA/postgresql.conf"
    # The default unix_socket_directories ('/var/run/postgresql') doesn't exist
    # inside the enroot container on Alps daint, so postgres fails to create
    # its socket lock file. /tmp is always writable.
    echo "unix_socket_directories = '/tmp'" >> "$PGDATA/postgresql.conf"

    # Start postgres (long -w timeout so WAL recovery can finish on restart
    # after a walltime/OOM kill — default 60s is not enough for 18GB+ of
    # vectors mid-CREATE-INDEX)
    pg_run "pg_ctl start -D '$PGDATA' -l '$PGDATA/logfile' -w -t 1200"

    # Wait for ready
    echo "Waiting for postgres..."
    for i in $(seq 1 1200); do
        if pg_run "pg_isready -q"; then
            echo "Postgres is ready."
            break
        fi
        sleep 1
    done

    # Create database and extensions
    pg_run "psql -U '$POSTGRES_USER' -d postgres -c \"CREATE DATABASE $POSTGRES_DB;\""
    pg_run "psql -U '$POSTGRES_USER' -d '$POSTGRES_DB' -c \"CREATE EXTENSION IF NOT EXISTS vector;\""
    pg_run "psql -U '$POSTGRES_USER' -d '$POSTGRES_DB' -c \"CREATE EXTENSION IF NOT EXISTS pg_prewarm;\""

    echo "=== Database initialized: $POSTGRES_DB ==="
else
    echo "=== Starting postgres (existing data) ==="
    pg_run "pg_ctl start -D '$PGDATA' -l '$PGDATA/logfile' -w -t 1200"

    echo "Waiting for postgres..."
    for i in $(seq 1 1200); do
        if pg_run "pg_isready -q"; then
            echo "Postgres is ready."
            break
        fi
        sleep 1
    done
fi

# --- Run the user's command ---
# If CMD is "postgres", just keep the container alive (postgres is already running in background).
# Otherwise, run the command, capture its exit code, then fall through to the
# EXIT trap so postgres gets stopped cleanly with `pg_ctl stop -m fast`.
#
# Important: do NOT use `exec "$@"` here. exec replaces this shell process,
# which discards the bash trap, so when the user command exits postgres is
# never asked to stop cleanly. SLURM then SIGKILLs the cgroup and leaves a
# stale postmaster.pid behind that breaks the next job's startup.
if [ "${1:-}" = "postgres" ]; then
    echo "Postgres running. PID: $(head -1 "$PGDATA/postmaster.pid")"
    # Keep container alive — wait for signal
    tail -f /dev/null &
    wait $!
else
    # Run without `set -e` aborting before we get a chance to exit cleanly,
    # so the EXIT trap fires and stops postgres regardless of user-cmd status.
    USER_EXIT=0
    "$@" || USER_EXIT=$?
    exit $USER_EXIT
fi
