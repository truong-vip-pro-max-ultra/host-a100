"""
Bearer API keys for the public /v1/* proxy (the API farm).

These are SEPARATE from the session password that gates the web UI: external
clients (Claude Code, Cline, the OpenAI SDK, …) authenticate to /v1/* with an
`Authorization: Bearer <key>` header, not a cookie. A key is a long random
token shown ONCE in full in the UI; we store it verbatim (it is a capability,
not a user password, and the whole DB is already owner-only).

The proxy ALWAYS requires a valid key, even when the web UI's password login is
disabled, because /v1/* is meant to be reached from the public internet through
the Cloudflare tunnel.
"""
import hmac
import secrets

from services import storage_service as db


def _new_token():
    # `sk-` prefix so it looks like (and pastes into) any OpenAI-style client.
    return "sk-hosta100-" + secrets.token_urlsafe(32)


def list_keys(include_revoked=False):
    sql = "SELECT * FROM api_keys"
    if not include_revoked:
        sql += " WHERE revoked=0"
    sql += " ORDER BY created_at DESC"
    rows = db.execute(sql, fetch="all") or []
    return [dict(r) for r in rows]


def create_key(label=""):
    """Mint a new key and return its full plaintext (shown once in the UI)."""
    token = _new_token()
    db.execute(
        "INSERT INTO api_keys (key, label, created_at, revoked) VALUES (?,?,?,0)",
        (token, (label or "").strip()[:128], db.now()), commit=True,
    )
    return token


def revoke_key(key_id):
    db.execute("UPDATE api_keys SET revoked=1 WHERE id=?", (key_id,), commit=True)


def delete_key(key_id):
    db.execute("DELETE FROM api_keys WHERE id=?", (key_id,), commit=True)


def ensure_default_key():
    """Make sure at least one usable key exists, so the API farm is reachable
    the moment the user opens the tab. Returns the plaintext of a freshly
    minted key, or None if a key already existed."""
    existing = db.execute(
        "SELECT COUNT(*) AS n FROM api_keys WHERE revoked=0", fetch="one")
    if existing and existing["n"]:
        return None
    return create_key("mặc định")


def verify(token):
    """Constant-time check that `token` is a live (non-revoked) key."""
    if not token:
        return False
    rows = db.execute(
        "SELECT key FROM api_keys WHERE revoked=0", fetch="all") or []
    ok = False
    for r in rows:
        # Compare against every key (don't early-return) to keep timing flat.
        if hmac.compare_digest(token, r["key"]):
            ok = True
    return ok


def mask(token):
    """A short, safe-to-display fingerprint of a key for the list view."""
    if not token or len(token) < 10:
        return "••••"
    return token[:11] + "…" + token[-4:]
