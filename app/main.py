# =============================================================================
#  SnapFind Studio - photo delivery by face
#  Copyright (c) 2026 Perarasu Boopathirajan. All rights reserved.
#  Author: Perarasu Boopathirajan
#  Unauthorized copying, redistribution or claiming authorship of this file,
#  via any medium, is prohibited without written permission. See LICENSE.
# =============================================================================
"""SnapFind - studio photo delivery with QR + face matching.

Admin uploads event photos -> faces are indexed -> guests scan the event QR,
take a selfie with the front camera, and get (and download) only their photos.
"""
import io
import os
import re
import secrets
import sqlite3
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from typing import List

import cv2
import numpy as np
import qrcode
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response)
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from starlette.middleware.sessions import SessionMiddleware

from .face_engine import MATCH_THRESHOLD, FaceEngine, get_engine

# ---------------------------------------------------------------- config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _pick_data_dir():
    if os.environ.get("DATA_DIR"):
        return os.environ["DATA_DIR"]
    # Hugging Face persistent storage is mounted at /data
    if os.path.isdir("/data") and os.access("/data", os.W_OK):
        return "/data/snapfind"
    return os.path.join(os.path.dirname(BASE_DIR), "data")


DATA_DIR = _pick_data_dir()
PHOTO_DIR = os.path.join(DATA_DIR, "photos")
THUMB_DIR = os.path.join(DATA_DIR, "thumbs")
DB_PATH = os.path.join(DATA_DIR, "snapfind.db")
for d in (PHOTO_DIR, THUMB_DIR):
    os.makedirs(d, exist_ok=True)

AUTHOR = "Perarasu Boopathirajan"
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
# No more public default password: if ADMIN_PASSWORD is not set, a random one is
# generated at startup and printed to the server log (HF: "Logs" tab).
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD") or ""
PASSWORD_IS_TEMP = not ADMIN_PASSWORD
if PASSWORD_IS_TEMP:
    ADMIN_PASSWORD = secrets.token_urlsafe(12)
    print("=" * 64, flush=True)
    print(f"[snapfind] ADMIN_PASSWORD not set. Temporary admin login for this run:", flush=True)
    print(f"[snapfind]   username: {ADMIN_USER}   password: {ADMIN_PASSWORD}", flush=True)
    print("[snapfind] Set ADMIN_PASSWORD to choose your own (it changes on every restart).", flush=True)
    print("=" * 64, flush=True)
SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
STUDIO_NAME = os.environ.get("STUDIO_NAME", "SnapFind Studio")
ON_HF = bool(os.environ.get("SPACE_ID"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "40"))
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}
MAX_SELFIE_MB = 15
LOGIN_MAX_FAILS = 10          # failed logins per IP ...
LOGIN_WINDOW = 15 * 60        # ... within this many seconds -> temporary lock

# ---------------------------------------------------------------- db
SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  slug TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  event_date TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS photos(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  uid TEXT UNIQUE NOT NULL,
  event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  orig_name TEXT,
  ext TEXT,
  width INTEGER, height INTEGER,
  face_count INTEGER DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS faces(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  event_id INTEGER NOT NULL,
  embedding BLOB NOT NULL,
  x REAL, y REAL, w REAL, h REAL
);
CREATE INDEX IF NOT EXISTS idx_faces_event ON faces(event_id);
CREATE INDEX IF NOT EXISTS idx_photos_event ON photos(event_id);
"""


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


with db() as _c:
    _c.executescript(SCHEMA)
    _c.execute("PRAGMA journal_mode=WAL")

# In-memory cache of each event's face matrix -> matching is one matrix multiply.
_cache_lock = threading.Lock()
_face_cache = {}  # event_id -> (photo_ids ndarray, embeddings ndarray)


def invalidate(event_id: int):
    with _cache_lock:
        _face_cache.pop(event_id, None)


def event_faces(event_id: int):
    with _cache_lock:
        if event_id in _face_cache:
            return _face_cache[event_id]
    with db() as c:
        rows = c.execute("SELECT photo_id, embedding FROM faces WHERE event_id=?", (event_id,)).fetchall()
    if rows:
        pids = np.array([r["photo_id"] for r in rows], dtype=np.int64)
        embs = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    else:
        pids, embs = np.zeros(0, dtype=np.int64), np.zeros((0, 128), dtype=np.float32)
    with _cache_lock:
        _face_cache[event_id] = (pids, embs)
    return pids, embs


# ---------------------------------------------------------------- app
app = FastAPI(title="SnapFind Studio", description=f"Photo delivery by face. (c) 2026 {AUTHOR}. All rights reserved.",
              contact={"name": AUTHOR})


@app.middleware("http")
async def add_author_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Author"] = AUTHOR
    return response


app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="snapfind_admin",
    max_age=60 * 60 * 12,
    same_site="none" if ON_HF else "lax",  # HF shows Spaces inside an iframe
    https_only=ON_HF,
)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
templates.env.globals["STUDIO_NAME"] = STUDIO_NAME
templates.env.globals["AUTHOR"] = AUTHOR


def csrf_token(request: Request) -> str:
    tok = request.session.get("csrf")
    if not tok:
        tok = secrets.token_urlsafe(32)
        request.session["csrf"] = tok
    return tok


def render(request: Request, name: str, **ctx):
    # only admins get a session/CSRF token; guests stay cookie-free
    ctx.setdefault("csrf", csrf_token(request) if is_admin(request) else "")
    return templates.TemplateResponse(request, name, ctx)


def is_admin(request: Request) -> bool:
    return request.session.get("admin") is True


def require_admin(request: Request, csrf: str = None):
    """Admin check + CSRF check for anything that changes data.

    The session cookie is SameSite=None on Hugging Face (iframe), so without a
    CSRF token any website could make a logged-in admin's browser create or
    delete events / upload photos.
    """
    if not is_admin(request):
        raise HTTPException(status_code=401, detail="Admin login required")
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        sent = csrf or request.headers.get("x-csrf-token") or ""
        expected = request.session.get("csrf") or ""
        if not expected or not secrets.compare_digest(sent, expected):
            raise HTTPException(status_code=403, detail="Invalid or missing CSRF token. Reload the page and try again.")


_login_fails = {}  # ip -> [timestamps]
_login_lock = threading.Lock()


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?"))


def login_blocked(ip: str) -> bool:
    now = time.time()
    with _login_lock:
        fails = [t for t in _login_fails.get(ip, []) if now - t < LOGIN_WINDOW]
        _login_fails[ip] = fails
        return len(fails) >= LOGIN_MAX_FAILS


def record_fail(ip: str):
    with _login_lock:
        _login_fails.setdefault(ip, []).append(time.time())


def public_base(request: Request) -> str:
    if os.environ.get("PUBLIC_URL"):
        return os.environ["PUBLIC_URL"].rstrip("/")
    if os.environ.get("SPACE_HOST"):  # set automatically on Hugging Face
        return "https://" + os.environ["SPACE_HOST"].split(",")[0]
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}"


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "event"
    return f"{s}-{secrets.token_hex(3)}"


def get_event(slug_or_id, by="slug"):
    with db() as c:
        row = c.execute(f"SELECT * FROM events WHERE {by}=?", (slug_or_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Event not found")
    return dict(row)


def photo_paths(uid: str, ext: str):
    return os.path.join(PHOTO_DIR, uid + ext), os.path.join(THUMB_DIR, uid + ".jpg")


def make_thumb(img: np.ndarray, path: str, max_side=640):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    t = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    cv2.imwrite(path, t, [cv2.IMWRITE_JPEG_QUALITY, 82])


# ================================================================ PUBLIC
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render(request, "home.html")


@app.post("/go")
def go(code: str = Form(...)):
    # accept a full link or just the code; keep only valid slug characters
    code = re.sub(r"[^a-z0-9-]", "", code.strip().rstrip("/").split("/")[-1].lower())[:80]
    return RedirectResponse(f"/e/{code}" if code else "/", status_code=303)


@app.get("/e/{slug}", response_class=HTMLResponse)
def guest_page(request: Request, slug: str):
    ev = get_event(slug)
    with db() as c:
        n = c.execute("SELECT COUNT(*) FROM photos WHERE event_id=?", (ev["id"],)).fetchone()[0]
    return render(request, "guest.html", ev=ev, photo_count=n)


@app.post("/api/e/{slug}/match")
async def match(slug: str, selfie: UploadFile = File(...)):
    ev = get_event(slug)
    data = await selfie.read(MAX_SELFIE_MB * 1024 * 1024 + 1)
    if len(data) > MAX_SELFIE_MB * 1024 * 1024:
        raise HTTPException(413, "Selfie too large")
    try:
        img = await run_in_threadpool(FaceEngine.load_image, data)
    except Exception:
        raise HTTPException(400, "Could not read that image")
    # CPU-heavy work runs in a worker thread so other guests aren't frozen
    face = await run_in_threadpool(lambda: get_engine().best_face(img))
    if face is None:
        return JSONResponse({"ok": False, "error": "no_face",
                             "message": "We couldn't see a face. Face the camera in good light and try again."})
    emb, _ = face
    pids, embs = event_faces(ev["id"])
    if len(pids) == 0:
        return {"ok": True, "matches": []}
    sims = embs @ emb
    best = {}
    for pid, s in zip(pids.tolist(), sims.tolist()):
        if s >= MATCH_THRESHOLD and s > best.get(pid, -1):
            best[pid] = s
    if not best:
        return {"ok": True, "matches": []}
    ids = sorted(best, key=best.get, reverse=True)
    with db() as c:
        q = f"SELECT id, uid, ext, orig_name, width, height FROM photos WHERE id IN ({','.join('?' * len(ids))})"
        rows = {r["id"]: dict(r) for r in c.execute(q, ids).fetchall()}
    out = []
    for pid in ids:
        r = rows.get(pid)
        if not r:
            continue
        out.append({
            "uid": r["uid"],
            "thumb": f"/t/{r['uid']}.jpg",
            "full": f"/p/{r['uid']}",
            "download": f"/p/{r['uid']}?dl=1",
            "w": r["width"], "h": r["height"],
            "score": round(best[pid], 3),
        })
    return {"ok": True, "matches": out}


@app.get("/t/{uid}.jpg")
def thumb(uid: str):
    if not re.fullmatch(r"[a-f0-9]{32}", uid):
        raise HTTPException(404)
    _, tp = photo_paths(uid, "")
    if not os.path.exists(tp):
        raise HTTPException(404)
    return FileResponse(tp, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/p/{uid}")
def full_photo(uid: str, dl: int = 0):
    if not re.fullmatch(r"[a-f0-9]{32}", uid):
        raise HTTPException(404)
    with db() as c:
        r = c.execute("SELECT p.*, e.name AS ev FROM photos p JOIN events e ON e.id=p.event_id WHERE uid=?",
                      (uid,)).fetchone()
    if not r:
        raise HTTPException(404)
    path, _ = photo_paths(uid, r["ext"])
    name = r["orig_name"] or (uid + r["ext"])
    if dl:
        return FileResponse(path, filename=name)
    return FileResponse(path)


@app.post("/api/e/{slug}/zip")
async def guest_zip(slug: str, request: Request):
    ev = get_event(slug)
    try:
        body = await request.json()
        raw = body.get("uids", []) if isinstance(body, dict) else []
    except Exception:
        raise HTTPException(400, "Invalid request")
    uids = [u for u in (raw if isinstance(raw, list) else []) if re.fullmatch(r"[a-f0-9]{32}", str(u))][:500]
    if not uids:
        raise HTTPException(400, "Nothing selected")
    with db() as c:
        q = f"SELECT uid, ext, orig_name FROM photos WHERE event_id=? AND uid IN ({','.join('?' * len(uids))})"
        rows = c.execute(q, [ev["id"], *uids]).fetchall()
    return zip_response(rows, f"{ev['slug']}-my-photos.zip")


def zip_response(rows, filename):
    # Build the ZIP in a temp file, not in RAM: "download all" for a big event
    # can be many GB and would crash a small server.
    fd, tmp_path = tempfile.mkstemp(suffix=".zip", dir=DATA_DIR)
    os.close(fd)
    used = set()
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED, allowZip64=True) as z:  # JPEGs don't compress further
        for r in rows:
            path, _ = photo_paths(r["uid"], r["ext"])
            if not os.path.exists(path):
                continue
            name = r["orig_name"] or (r["uid"] + r["ext"])
            base, ext = os.path.splitext(name)
            i = 1
            while name in used:
                name = f"{base}_{i}{ext}"
                i += 1
            used.add(name)
            z.write(path, name)
    return FileResponse(tmp_path, media_type="application/zip", filename=filename,
                        background=BackgroundTask(os.remove, tmp_path))


# ================================================================ ADMIN
@app.get("/admin/login", response_class=HTMLResponse)
def login_page(request: Request):
    if is_admin(request):
        return RedirectResponse("/admin", status_code=303)
    return render(request, "login.html", error=None, temp_pw=PASSWORD_IS_TEMP)


@app.post("/admin/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = client_ip(request)
    if login_blocked(ip):
        return render(request, "login.html", error="Too many failed attempts. Try again in 15 minutes.",
                      temp_pw=PASSWORD_IS_TEMP)
    if secrets.compare_digest(username.encode(), ADMIN_USER.encode()) and \
            secrets.compare_digest(password.encode(), ADMIN_PASSWORD.encode()):
        request.session.clear()  # fresh session (and fresh CSRF token) on login
        request.session["admin"] = True
        return RedirectResponse("/admin", status_code=303)
    record_fail(ip)
    time.sleep(0.8)
    return render(request, "login.html", error="Wrong username or password", temp_pw=PASSWORD_IS_TEMP)


@app.post("/admin/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/admin/login", status_code=303)


@app.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request):
    if not is_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    with db() as c:
        events = [dict(r) for r in c.execute("""
            SELECT e.*, COUNT(DISTINCT p.id) AS photos, COALESCE(SUM(p.face_count),0) AS faces
            FROM events e LEFT JOIN photos p ON p.event_id=e.id
            GROUP BY e.id ORDER BY e.created_at DESC""").fetchall()]
        totals = c.execute("SELECT (SELECT COUNT(*) FROM events), (SELECT COUNT(*) FROM photos), "
                           "(SELECT COUNT(*) FROM faces)").fetchone()
    return render(request, "admin.html", events=events,
                  totals={"events": totals[0], "photos": totals[1], "faces": totals[2]},
                  persistent=DATA_DIR.startswith("/data"), on_hf=ON_HF)


@app.post("/admin/events")
def create_event(request: Request, name: str = Form(...), event_date: str = Form(""), csrf: str = Form("")):
    require_admin(request, csrf)
    name = name.strip()[:120] or "Untitled event"
    slug = slugify(name)
    with db() as c:
        cur = c.execute("INSERT INTO events(slug,name,event_date,created_at) VALUES(?,?,?,?)",
                        (slug, name, event_date.strip()[:20], time.time()))
        eid = cur.lastrowid
    return RedirectResponse(f"/admin/events/{eid}", status_code=303)


@app.get("/admin/events/{eid}", response_class=HTMLResponse)
def admin_event(request: Request, eid: int):
    if not is_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    ev = get_event(eid, by="id")
    with db() as c:
        photos = [dict(r) for r in c.execute(
            "SELECT * FROM photos WHERE event_id=? ORDER BY created_at DESC", (eid,)).fetchall()]
    link = f"{public_base(request)}/e/{ev['slug']}"
    return render(request, "admin_event.html", ev=ev, photos=photos, link=link,
                  face_total=sum(p["face_count"] for p in photos))


@app.get("/admin/events/{eid}/qr.png")
def event_qr(request: Request, eid: int, size: int = 12):
    require_admin(request)
    ev = get_event(eid, by="id")
    link = f"{public_base(request)}/e/{ev['slug']}"
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=max(4, min(size, 30)), border=3)
    qr.add_data(link)
    qr.make(fit=True)
    buf = io.BytesIO()
    qr.make_image(fill_color="#111111", back_color="white").save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"Content-Disposition": f'inline; filename="{ev["slug"]}-qr.png"'})


@app.get("/admin/events/{eid}/poster", response_class=HTMLResponse)
def event_poster(request: Request, eid: int):
    require_admin(request)
    ev = get_event(eid, by="id")
    return render(request, "poster.html", ev=ev, link=f"{public_base(request)}/e/{ev['slug']}")


def _process_photo(eid: int, name: str, ext: str, data: bytes) -> dict:
    """Decode, store, thumbnail and index one photo (runs in a worker thread)."""
    try:
        img = FaceEngine.load_image(data)
    except Exception:
        return {"name": name, "ok": False, "error": "Not a readable image"}
    uid = uuid.uuid4().hex
    if ext in (".heic", ".heif", ".bmp", ".webp"):  # store a universally viewable JPEG
        ext = ".jpg"
        name = os.path.splitext(name)[0] + ".jpg"
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return {"name": name, "ok": False, "error": "Could not convert image"}
        data = enc.tobytes()
    path, tpath = photo_paths(uid, ext)
    try:
        with open(path, "wb") as out:
            out.write(data)
        make_thumb(img, tpath)
        found = get_engine().faces(img)
        h, w = img.shape[:2]
        with db() as c:
            cur = c.execute("INSERT INTO photos(uid,event_id,orig_name,ext,width,height,face_count,created_at) "
                            "VALUES(?,?,?,?,?,?,?,?)", (uid, eid, name, ext, w, h, len(found), time.time()))
            pid = cur.lastrowid
            c.executemany("INSERT INTO faces(photo_id,event_id,embedding,x,y,w,h) VALUES(?,?,?,?,?,?,?)",
                          [(pid, eid, e.tobytes(), *b) for e, b in found])
    except Exception as e:  # don't leave orphan files behind
        for p in (path, tpath):
            if os.path.exists(p):
                os.remove(p)
        return {"name": name, "ok": False, "error": f"Processing failed: {e}"}
    return {"name": name, "ok": True, "id": pid, "uid": uid, "faces": len(found), "thumb": f"/t/{uid}.jpg"}


@app.post("/api/admin/events/{eid}/photos")
async def upload_photos(request: Request, eid: int, files: List[UploadFile] = File(...)):
    require_admin(request)
    get_event(eid, by="id")
    limit = MAX_UPLOAD_MB * 1024 * 1024
    results = []
    for f in files:
        name = os.path.basename((f.filename or "photo.jpg").replace("\\", "/"))[:150] or "photo.jpg"
        ext = os.path.splitext(name)[1].lower() or ".jpg"
        if ext not in ALLOWED_EXT:
            results.append({"name": name, "ok": False, "error": "Unsupported file type"})
            continue
        data = await f.read(limit + 1)
        if len(data) > limit:
            results.append({"name": name, "ok": False, "error": f"Larger than {MAX_UPLOAD_MB} MB"})
            continue
        results.append(await run_in_threadpool(_process_photo, eid, name, ext, data))
    invalidate(eid)
    return {"results": results}


@app.delete("/api/admin/photos/{pid}")
def delete_photo(request: Request, pid: int):
    require_admin(request)
    with db() as c:
        r = c.execute("SELECT * FROM photos WHERE id=?", (pid,)).fetchone()
        if not r:
            raise HTTPException(404)
        c.execute("DELETE FROM faces WHERE photo_id=?", (pid,))
        c.execute("DELETE FROM photos WHERE id=?", (pid,))
    for p in photo_paths(r["uid"], r["ext"]):
        if os.path.exists(p):
            os.remove(p)
    invalidate(r["event_id"])
    return {"ok": True}


@app.post("/admin/events/{eid}/delete")
def delete_event(request: Request, eid: int, csrf: str = Form("")):
    require_admin(request, csrf)
    with db() as c:
        rows = c.execute("SELECT uid, ext FROM photos WHERE event_id=?", (eid,)).fetchall()
        c.execute("DELETE FROM faces WHERE event_id=?", (eid,))
        c.execute("DELETE FROM photos WHERE event_id=?", (eid,))
        c.execute("DELETE FROM events WHERE id=?", (eid,))
    for r in rows:
        for p in photo_paths(r["uid"], r["ext"]):
            if os.path.exists(p):
                os.remove(p)
    invalidate(eid)
    return RedirectResponse("/admin", status_code=303)


@app.get("/admin/events/{eid}/download.zip")
def admin_zip(request: Request, eid: int):
    require_admin(request)
    ev = get_event(eid, by="id")
    with db() as c:
        rows = c.execute("SELECT uid, ext, orig_name FROM photos WHERE event_id=? ORDER BY created_at",
                         (eid,)).fetchall()
    return zip_response(rows, f"{ev['slug']}-all-photos.zip")


@app.get("/healthz")
def health():
    return {"ok": True, "author": AUTHOR}


@app.on_event("startup")
def _warm():
    # load models in the background so the first request isn't slow
    threading.Thread(target=get_engine, daemon=True).start()
