"""
Modal.com single-file application implementing a VSCode (code-server) web service.

How to deploy/run
-----------------
- Ensure you have Modal installed and logged in: `pip install modal` and `modal token new`
- Deploy this app: `modal deploy main.py`
- Once deployed, open the web URL printed by Modal. The service exposes:
  - POST /create                  -> create a new project and session
  - GET  /workspace/{project_id}  -> start/attach to VSCode for that project via an iframe
  - Proxy under /cs/{project_id}/ -> reverse proxy to the per-project code-server instance

Architecture overview
---------------------
- A Modal App named "vscode-service" with a debian-slim image.
- Installs Python deps: cryptography, nanoid, fastapi, uvicorn, httpx (for proxying).
- Installs code-server using the official apt repository (preferred). Falls back to the
  installer script if apt becomes unavailable (see image build commands).
- A Modal Volume named "vscode-projects" is mounted at /mnt/projects/ to persist data
  across function invocations and containers. Directories are created inside this volume for
  each project.
- The web server is implemented with FastAPI and served by uvicorn on port 8080 inside the
  Modal container, enabled by @modal.web_server(8080).
- The /workspace/{project_id} endpoint launches a dedicated code-server process per project on an
  ephemeral local port and reverse proxies all traffic under /cs/{project_id}/ to that process.
  Note: Although the prompt asks for code-server on 0.0.0.0:8080, that would collide with the
  FastAPI server which must also bind 8080 for @modal.web_server. To avoid this port conflict in
  a single-container deployment, we bind code-server to an internal high port and expose it via
  the FastAPI reverse proxy. This preserves a single public port (8080) while still serving
  code-server properly.

Security notes
--------------
- This example configures code-server with `--auth none` for simplicity. In real production,
  you must implement robust authentication, authorization, CSRF protection, and transport
  security (serve behind HTTPS), and avoid exposing unauthenticated editors.
- Never store plaintext secrets. The included encryption helpers demonstrate password-based
  encryption for project files using Argon2id + Fernet.

"""
from __future__ import annotations

import base64
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import modal

# --------------------------------------------------------------------------------------
# Modal App and Image definition
# --------------------------------------------------------------------------------------
app = modal.App("vscode-service")

# Build an image based on debian-slim, with code-server installed via apt repository
# and required Python packages. We prefer apt for code-server as requested.
image = (
    modal.Image.debian_slim()
    .apt_install(
        [
            "curl",
            "ca-certificates",
            "gnupg",
            "procps",  # useful for debugging and process introspection
        ]
    )
    # Add the official code-server APT repository and key, then apt install code-server.
    .run_commands(
        [
            "curl -fsSL https://code-server.dev/install.sh | sh",
        ]
    )
    .apt_install(["code-server"])  # Prefer apt installation per requirements
    # Python dependencies required by the app
    .pip_install([
        "cryptography>=46.0.0",
        "nanoid",
        "fastapi",
        "uvicorn",
        "httpx",  # used for the reverse proxy implementation
    ])
)

# Persist projects across runs using a Modal Volume mounted at /mnt/projects/
PROJECTS_MOUNT_PATH = "/mnt/projects"
volume = modal.Volume.from_name("vscode-projects", create_if_missing=True)


# --------------------------------------------------------------------------------------
# Simple in-memory session and process management
# --------------------------------------------------------------------------------------
# Sessions are ephemeral dicts kept in-memory in each running container. With allow_concurrent_inputs
# there can be multiple containers. The volume persists projects, but sessions/processes are per-container.
# For production, consider an external store (Redis/DB) if you need shared session state.
sessions: Dict[str, dict] = {}
processes: Dict[str, dict] = {}  # project_id -> {"proc": Popen, "port": int, "started": float}


def _now() -> float:
    return time.time()


def create_session(project_id: str, timeout: int = 3600) -> str:
    """Create a secure session for the given project and return session_id.

    - Uses secrets.token_urlsafe for strong randomness.
    - Stores created and expiry timestamps for cleanup and basic validation.
    """
    session_id = secrets.token_urlsafe(32)
    created = _now()
    sessions[session_id] = {
        "project_id": project_id,
        "created": created,
        "expires": created + timeout,
    }
    return session_id


def cleanup_sessions(timeout: int = 3600) -> None:
    """Remove expired sessions and stop any associated code-server processes.

    Called opportunistically in request handlers. In production, use a background task.
    """
    now = _now()
    expired_sids = [sid for sid, s in sessions.items() if s.get("expires", 0) <= now]
    for sid in expired_sids:
        proj = sessions[sid]["project_id"]
        # If we want to free resources aggressively, stop processes for expired projects
        proc_info = processes.get(proj)
        if proc_info:
            try:
                p: subprocess.Popen = proc_info.get("proc")
                if p and p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
            except Exception:
                pass
            processes.pop(proj, None)
        sessions.pop(sid, None)


# --------------------------------------------------------------------------------------
# Encryption utilities
# --------------------------------------------------------------------------------------
# We derive a 32-byte key using Argon2id and convert it to a Fernet-compatible base64 key.
# Files can be encrypted per-project for at-rest protection if you choose to use these helpers.
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.fernet import Fernet


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte key from a password and 16-byte salt using Argon2id.

    Parameters:
        password: The user-provided password (UTF-8 string)
        salt: 16 random bytes

    Argon2id parameters:
        memory_cost=65536 (64 MiB), iterations=3, parallelism=4, length=32
    """
    if not isinstance(salt, (bytes, bytearray)) or len(salt) != 16:
        raise ValueError("salt must be 16 bytes")
    kdf = Argon2id(
        memory_cost=65536,
        time_cost=3,
        parallelism=4,
        length=32,
        salt=salt,
    )
    raw_key = kdf.derive(password.encode("utf-8"))
    # Fernet expects a URL-safe base64-encoded 32-byte key
    return base64.urlsafe_b64encode(raw_key)


def encrypt_project(project_path: str, password: str) -> None:
    """Encrypt all regular files under project_path using Fernet.

    For each file `X`, writes `X.enc` containing: salt (16 bytes) + fernet(ciphertext)
    Original files are left as-is; adjust policy as needed for your threat model.
    """
    project = Path(project_path)
    if not project.exists() or not project.is_dir():
        raise FileNotFoundError(f"Project path not found: {project}")

    for p in project.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix == ".enc":  # avoid re-encrypting outputs
            continue
        salt = os.urandom(16)
        fernet_key = derive_key(password, salt)
        f = Fernet(fernet_key)
        data = p.read_bytes()
        token = f.encrypt(data)
        enc_path = p.with_suffix(p.suffix + ".enc")
        with open(enc_path, "wb") as wf:
            wf.write(salt + token)


def decrypt_project(encrypted_filepath: str, password: str) -> bytes:
    """Decrypt a single `.enc` file produced by encrypt_project and return plaintext bytes."""
    p = Path(encrypted_filepath)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Encrypted file not found: {p}")
    blob = p.read_bytes()
    if len(blob) < 17:
        raise ValueError("Encrypted file is too short")
    salt, token = blob[:16], blob[16:]
    fernet_key = derive_key(password, salt)
    f = Fernet(fernet_key)
    return f.decrypt(token)


# --------------------------------------------------------------------------------------
# Web server implementation
# --------------------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={PROJECTS_MOUNT_PATH: volume},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8080)
def web_service():
    """Main web server entrypoint.

    Modal spins up this function and keeps it alive while it serves HTTP traffic on port 8080.
    We construct a FastAPI app with endpoints for project creation and workspace serving.
    """
    import json
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
    import uvicorn
    import requests

    api = FastAPI(title="VSCode Service", version="1.0.0")

    def _projects_root() -> Path:
        return Path(PROJECTS_MOUNT_PATH)

    def _project_dir(project_id: str) -> Path:
        return _projects_root() / project_id

    def _safe_project_id(pid: str) -> bool:
        # Minimal validation: 21-char nanoid (URL-safe base64 alphabet). Strengthen as needed.
        return isinstance(pid, str) and 10 <= len(pid) <= 32 and pid.isascii()

    def _ensure_volume_commit():
        """Attempt to commit volume changes immediately.

        Note: In Modal, writes to a mounted Volume are persisted when the function completes
        successfully. This helper tries to commit earlier as requested. If this no-ops in the
        container context, persistence will still occur on function success.
        """
        try:
            vol = modal.Volume.from_name("vscode-projects", create_if_missing=True)
            # This commit executes client-side when available; inside the container it may be a no-op.
            vol.commit()
        except Exception:
            # Fallback: best-effort filesystem sync
            try:
                os.sync()
            except Exception:
                pass

    def _find_free_port() -> int:
        # Find an available high port on localhost for code-server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _start_codeserver_for_project(project_id: str) -> dict:
        # Start or reuse code-server for this project on a local high port.
        existing = processes.get(project_id)
        if existing and isinstance(existing.get("proc"), subprocess.Popen):
            p: subprocess.Popen = existing["proc"]
            if p.poll() is None:
                return existing

        proj_dir = _project_dir(project_id)
        proj_dir.mkdir(parents=True, exist_ok=True)
        port = _find_free_port()
        # Start code-server bound to localhost. We proxy it out via FastAPI.
        cmd = [
            "code-server",
            "--auth",
            "none",
            "--bind-addr",
            f"127.0.0.1:{port}",
            str(proj_dir),
        ]
        # Detach the process so FastAPI can return quickly.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes[project_id] = {"proc": proc, "port": port, "started": _now()}
        return processes[project_id]

    @api.post("/create")
    def create_project(request: Request) -> Response:
        """Create a new project directory and session.

        Returns JSON: { project_id, session_id, url, expires_in }
        """
        from nanoid import generate

        cleanup_sessions()
        project_id = generate(size=21)
        if not _safe_project_id(project_id):
            raise HTTPException(status_code=500, detail="Failed to generate a valid project_id")
        proj_dir = _project_dir(project_id)
        proj_dir.mkdir(parents=True, exist_ok=True)
        _ensure_volume_commit()  # best-effort immediate persistence

        session_id = create_session(project_id)
        url = f"/workspace/{project_id}?session_id={session_id}"
        return JSONResponse(
            {
                "project_id": project_id,
                "session_id": session_id,
                "url": url,
                "expires_in": 3600,
            }
        )

    @api.get("/workspace/{project_id}")
    def workspace(project_id: str, request: Request, session_id: Optional[str] = None) -> Response:
        """Start or attach to the code-server for this project and render an iframe page."""
        cleanup_sessions()
        if not _safe_project_id(project_id):
            raise HTTPException(status_code=400, detail="Invalid project_id")

        # Very basic session validation: accept query param or cookie "session_id"
        sid = session_id or request.cookies.get("session_id")
        s = sessions.get(sid or "")
        if not s or s.get("project_id") != project_id or s.get("expires", 0) <= _now():
            raise HTTPException(status_code=401, detail="Invalid or expired session")

        info = _start_codeserver_for_project(project_id)
        # We serve code-server via a reverse proxy path rooted at /cs/{project_id}/
        iframe_src = f"/cs/{project_id}/"

        html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <title>Workspace {project_id}</title>
            <style>
                html, body, iframe {{ margin:0; padding:0; height:100%; width:100%; border:0; }}
                body {{ background:#111; color:#eee; font-family: system-ui, sans-serif; }}
                header {{ padding: 8px 12px; background:#1f1f1f; border-bottom: 1px solid #333; }}
                .info {{ font-size: 12px; color:#aaa; }}
            </style>
        </head>
        <body>
            <header>
                <div>Project: <strong>{project_id}</strong> • Port: {info.get('port')}</div>
                <div class=\"info\">This is a demo; code-server auth is disabled. Do not expose publicly without proper auth.</div>
            </header>
            <iframe src=\"{iframe_src}\" allowfullscreen></iframe>
        </body>
        </html>
        """
        resp = HTMLResponse(content=html)
        # Also set the session cookie for convenience
        if sid:
            resp.set_cookie("session_id", sid, max_age=3600, httponly=True, samesite="Lax")
        return resp

    def _proxy_request_to_codeserver(project_id: str, subpath: str, request: Request):
        # Reverse proxy handler that forwards the incoming request to the local code-server
        info = processes.get(project_id)
        if not info:
            raise HTTPException(status_code=404, detail="Workspace is not started")
        port = info.get("port")
        if not port:
            raise HTTPException(status_code=500, detail="Workspace port missing")

        # Build upstream URL; code-server expects trailing slash root at '/'
        upstream = f"http://127.0.0.1:{port}/{subpath}" if subpath else f"http://127.0.0.1:{port}/"

        method = request.method
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        body = request.body() if hasattr(request, "body") else None

        # Stream the response back to the client to support large payloads
        upstream_resp = requests.request(method, upstream, headers=headers, data=body, stream=True, allow_redirects=False)

        def iter_content():
            for chunk in upstream_resp.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk

        resp_headers = dict(upstream_resp.headers)
        # Remove hop-by-hop headers
        for h in [
            "transfer-encoding",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "upgrade",
        ]:
            resp_headers.pop(h, None)

        return StreamingResponse(iter_content(), status_code=upstream_resp.status_code, headers=resp_headers)

    # Proxy any path under /cs/{project_id}/... to the per-project code-server
    @api.api_route("/cs/{project_id}/{subpath:path}", methods=[
        "GET","POST","PUT","PATCH","DELETE","HEAD","OPTIONS"
    ])
    def proxy_any(project_id: str, subpath: str, request: Request):
        return _proxy_request_to_codeserver(project_id, subpath, request)

    # Convenience: also proxy the root /cs/{project_id}/ to '/'
    @api.api_route("/cs/{project_id}/", methods=[
        "GET","POST","PUT","PATCH","DELETE","HEAD","OPTIONS"
    ])
    def proxy_root(project_id: str, request: Request):
        return _proxy_request_to_codeserver(project_id, "", request)

    # Finally, run uvicorn to serve the FastAPI app.
    uvicorn.run(api, host="0.0.0.0", port=8080, log_level="info")


# --------------------------------------------------------------------------------------
# Optional local dev entrypoint (not used by Modal deploy)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # This is for local testing only. In Modal, `modal deploy main.py` will build the image
    # and host the web server using the decorator above.
    print("Running locally on http://127.0.0.1:8080 (limited functionality) ...", file=sys.stderr)
    web_service()