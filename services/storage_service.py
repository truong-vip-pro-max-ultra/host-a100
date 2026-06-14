"""
SQLite access layer and disk-usage reporting.

A single shared SQLite database stores models, envs and jobs metadata. Each
call opens its own short-lived connection (check_same_thread is irrelevant then)
which is the simplest correct approach for a multi-threaded single process.
"""
import shutil
import sqlite3
import threading
import time

import config

# Serialize writes across background threads. SQLite handles concurrency, but a
# process-wide lock keeps things simple and avoids 'database is locked' under
# the light write load this platform produces.
_write_lock = threading.Lock()


def connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    """Create tables if they do not yet exist."""
    config.ensure_dirs()
    with _write_lock, connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS models (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT UNIQUE NOT NULL,
                path       TEXT NOT NULL,
                size       INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS envs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT UNIQUE NOT NULL,
                path       TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT UNIQUE NOT NULL,
                path       TEXT NOT NULL,
                main_file  TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id   INTEGER,
                env_id     INTEGER,
                project_id INTEGER,
                main_file  TEXT,
                run_mode   TEXT NOT NULL DEFAULT 'runner',
                name       TEXT,
                status     TEXT NOT NULL DEFAULT 'queued',
                progress   INTEGER NOT NULL DEFAULT 0,
                logs_path  TEXT,
                result_path TEXT,
                output_path TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE SET NULL,
                FOREIGN KEY (env_id) REFERENCES envs(id) ON DELETE SET NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
            );

            -- API farm: each row is one long-running OpenAI-compatible LLM
            -- server (llama.cpp) dispatched to a GPU compute node via sbatch.
            -- The Flask app on the login node reverse-proxies /v1/* to
            -- http://<node>:<port>, which the server writes into endpoint.json.
            CREATE TABLE IF NOT EXISTS servers (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                served_name   TEXT NOT NULL,
                model_id      INTEGER,
                env_id        INTEGER,
                engine        TEXT NOT NULL DEFAULT 'llamacpp',
                status        TEXT NOT NULL DEFAULT 'queued',
                slurm_job_id  TEXT,
                node          TEXT,
                port          INTEGER,
                gpu_model     TEXT,
                n_gpu_layers  INTEGER,
                n_ctx         INTEGER,
                chat_format   TEXT,
                extra_args    TEXT,
                time_limit    TEXT,
                auto_resubmit INTEGER NOT NULL DEFAULT 1,
                logs_path     TEXT,
                created_at    REAL NOT NULL,
                stopped_at    REAL,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE SET NULL,
                FOREIGN KEY (env_id)   REFERENCES envs(id)   ON DELETE SET NULL
            );

            -- Bearer API keys that gate the public /v1/* proxy. Independent of
            -- the session password (clients send Authorization: Bearer <key>).
            CREATE TABLE IF NOT EXISTS api_keys (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key        TEXT UNIQUE NOT NULL,
                label      TEXT,
                created_at REAL NOT NULL,
                revoked    INTEGER NOT NULL DEFAULT 0
            );

            -- Tools / Clone giọng nói: one long-running OmniVoice TTS server on a
            -- GPU compute node (sbatch), same lifecycle as the API-farm `servers`.
            -- It writes <node>:<port> into endpoint.json; the login-node app POSTs
            -- synthesis requests to http://<node>:<port>/synthesize.
            CREATE TABLE IF NOT EXISTS voice_servers (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                env_id        INTEGER,
                model_id      TEXT,            -- HF model id (resolved from cache)
                status        TEXT NOT NULL DEFAULT 'queued',
                slurm_job_id  TEXT,
                node          TEXT,
                port          INTEGER,
                gpu_model     TEXT,
                time_limit    TEXT,
                auto_resubmit INTEGER NOT NULL DEFAULT 1,
                logs_path     TEXT,
                created_at    REAL NOT NULL,
                stopped_at    REAL,
                FOREIGN KEY (env_id) REFERENCES envs(id) ON DELETE SET NULL
            );

            -- A cloned / zero-shot named voice: a (preprocessed) reference clip on
            -- the shared FS plus its transcript. Synthesis conditions on this.
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT UNIQUE NOT NULL,
                ref_audio  TEXT NOT NULL,      -- abs path to the stored reference wav
                ref_text   TEXT,
                language   TEXT NOT NULL DEFAULT 'vi',
                created_at REAL NOT NULL
            );

            -- One narration job: script text → MP3 (+SRT). Runs on the login node
            -- (chunk + ffmpeg) and calls the GPU server per chunk.
            CREATE TABLE IF NOT EXISTS voice_jobs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT,
                status      TEXT NOT NULL DEFAULT 'queued',
                progress    INTEGER NOT NULL DEFAULT 0,
                stage       TEXT,
                params      TEXT,              -- JSON of the synth knobs
                output_path TEXT,             -- final MP3
                srt_path    TEXT,
                logs_path   TEXT,             -- the job working dir
                error       TEXT,
                created_at  REAL NOT NULL,
                finished_at REAL
            );
            """
        )
        _migrate(conn)


def _migrate(conn):
    """Add columns introduced after the first release to pre-existing DBs."""
    have = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    additions = {
        "project_id": "INTEGER",
        "main_file": "TEXT",
        "run_mode": "TEXT NOT NULL DEFAULT 'runner'",
        "output_path": "TEXT",
    }
    for col, decl in additions.items():
        if col not in have:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")


def execute(sql, params=(), *, commit=False, fetch=None):
    """
    Run a query. fetch in {None, 'one', 'all'}. Writes take the global lock.
    Returns lastrowid for commit writes, or rows for reads.
    """
    if commit:
        with _write_lock, connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.lastrowid
    with connect() as conn:
        cur = conn.execute(sql, params)
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        return None


def now():
    return time.time()


def disk_usage():
    """Return (used, total, free) bytes for the data filesystem."""
    config.ensure_dirs()
    try:
        usage = shutil.disk_usage(config.DATA_DIR)
        return usage.used, usage.total, usage.free
    except OSError:
        return 0, 0, 0
