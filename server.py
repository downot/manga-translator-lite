import http.server
import socketserver
import json
import os
import secrets
import hashlib
import urllib.parse
import sys
import time
import threading
import queue
import asyncio
import logging
import shutil
from pathlib import Path

# Add project root to sys.path so manga_translator_lite can be imported *if present*.
sys.path.append(str(Path(__file__).parent.absolute()))

# Generic, transport-agnostic base logic shared with downstream editor servers.
# It's a plain top-level module (not under manga_translator_lite), so importing it
# never pulls the package's heavy ML deps — the editor still starts standalone.
import server_common as sc

# Customizable tool name (tab title + header brand in editor.html). Empty = default.
APP_NAME = os.environ.get("MTL_APP_NAME", "").strip()
_EDITOR_NAME_MARKER = 'window.MTL_APP_NAME = "Manga Translator";'

# The editor itself (view/edit translations, pages, images, story) runs WITHOUT the
# manga_translator_lite package. Only the pipeline runner (/api/pipeline/run) needs it,
# so those imports are deferred to that endpoint — letting the editor start standalone.


class LogQueueHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):
        self.q.put(self.format(record))

class TokenManager:
    def __init__(self, work_dir, secret=None):
        self.work_dir = Path(work_dir)
        self.secret = secret or secrets.token_hex(16)
        self.super_token = hashlib.sha256(("super:" + self.secret).encode()).hexdigest()[:16]
        self.tokens = {} # token -> task_name
        self.tasks = {} # task_name -> token
        self.refresh()

    def refresh(self):
        if not self.work_dir.exists():
            return
        for item in self.work_dir.iterdir():
            if item.is_dir() and (item / 'pages.json').exists():
                task_name = item.name
                if task_name not in self.tasks:
                    token = hashlib.sha256((task_name + self.secret).encode()).hexdigest()[:16]
                    self.tokens[token] = task_name
                    self.tasks[task_name] = token

    def get_task_path(self, token):
        task_name = self.tokens.get(token)
        if task_name:
            return self.work_dir / task_name
        return None

    def get_all_links(self, base_url):
        links = []
        for token, task_name in self.tokens.items():
            links.append((task_name, f"{base_url}?t={token}"))
        return links

    def is_super_token(self, token):
        return bool(token) and secrets.compare_digest(token, self.super_token)

    def get_all_tasks(self):
        self.refresh()
        return [
            {"name": task_name, "token": token, "perm": "edit"}
            for task_name, token in sorted(self.tasks.items())
        ]

_COLLAB_LOCK = threading.Lock()
_SAVE_LOCK = threading.Lock()
_LISTENERS = {}      # task_name -> list of queue.Queue
_ACTIVE_USERS = {}   # task_name -> username -> dict of collab details
_CHAT_HISTORIES = {} # task_name -> list of chat events (capped at 30)


def broadcast_event(task_name, event):
    with _COLLAB_LOCK:
        queues = _LISTENERS.get(task_name, [])
        for q in queues:
            q.put(event)


def broadcast_presence(task_name):
    with _COLLAB_LOCK:
        users_info = []
        if task_name in _ACTIVE_USERS:
            for username, info in _ACTIVE_USERS[task_name].items():
                users_info.append({
                    "username": username,
                    "name": info["name"],
                    "avatar": info["avatar"],
                    "page": info["page"],
                    "block": info["block"],
                    "lang": info.get("lang")
                })
        event = {
            "type": "presence",
            "users": users_info
        }
    broadcast_event(task_name, event)


def broadcast_save(task_name, lang, mtime):
    event = {
        "type": "save",
        "lang": lang,
        "mtime": mtime
    }
    broadcast_event(task_name, event)


class EditorHandler(http.server.BaseHTTPRequestHandler):
    token_manager = None
    root_dir = Path(".")
    logger = None
    # Serialize pipeline runs across the threading server. The per-request log
    # handler is attached to the ROOT logger, so concurrent runs would otherwise
    # leak each other's log lines into both SSE streams (and contend for the GPU).
    pipeline_lock = threading.Lock()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        token = params.get('t', [None])[0]

        is_api = "/api/" in parsed.path or parsed.path.startswith("api/")

        if not is_api:
            self.serve_editor_html()
            return

        # Global (task-independent) SFX dictionary, kept next to editor.html. It loads at
        # editor startup before any task/token exists, so it bypasses the token gate.
        if parsed.path.endswith("/api/sfx-dict"):
            self.serve_sfx_dict()
            return

        if parsed.path.endswith("/api/tasks"):
            super_token = params.get('super', [None])[0]
            if not self.token_manager.is_super_token(super_token):
                self.send_error(403, "Super token required")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.token_manager.get_all_tasks()).encode())
            return

        if not token:
            self.send_error(403, "Token required")
            return

        # Pick up tasks created after startup (e.g. merged_* folders made by the
        # editor) without requiring a server restart.
        self.token_manager.refresh()
        task_path = self.token_manager.get_task_path(token)
        if not task_path:
            self.send_error(404, "Invalid token or task not found")
            return

        if parsed.path.endswith("/api/data"):
            self.serve_file(task_path / "pages.json", "application/json")
        elif parsed.path.endswith("/api/translations"):
            lang = params.get('lang', [None])[0]
            if lang:
                if not sc.is_safe_lang(lang):
                    self.send_error(400, "Invalid language format")
                    return
                trans_path = task_path / "translations" / f"{lang}.json"
                if trans_path.exists():
                    mtime = os.path.getmtime(trans_path)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("X-Translation-Mtime", str(mtime))
                    self.send_header("Content-Length", trans_path.stat().st_size)
                    self.end_headers()
                    with open(trans_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("X-Translation-Mtime", "0")
                    self.send_header("Content-Length", 2)
                    self.end_headers()
                    self.wfile.write(b"{}")
            else:
                self.send_error(400, "Language required")
        elif parsed.path.endswith("/api/collab/stream"):
            user = params.get('user', [None])[0] or "Anonymous"
            task_name = self.token_manager.tokens.get(token)
            if not task_name:
                self.send_error(404, "Task not found")
                return
            self.handle_collab_stream(task_name, user)
            return
        elif parsed.path.endswith("/api/story"):
            story_path = task_path / "story.txt"
            if story_path.exists():
                self.serve_file(story_path, "text/plain")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"")
        elif parsed.path.endswith("/api/image"):
            img_name = params.get('name', [None])[0]
            if img_name:
                ext = os.path.splitext(img_name)[1].lower()
                # webp is a first-class clean/output format and must be allowed,
                # else the main canvas image 403s while its thumbnail still loads.
                CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg",
                                 ".jpeg": "image/jpeg", ".webp": "image/webp",
                                 ".gif": "image/gif", ".bmp": "image/bmp"}
                if "/" in img_name or "\\" in img_name or ext not in CONTENT_TYPES:
                    self.send_error(403, "Invalid image name")
                    return
                # Check different possible image locations
                for sub in ["clean", "clean_v2"]:
                    p = task_path / sub / img_name
                    if p.exists():
                        self.serve_file(p, CONTENT_TYPES[ext])
                        return
                self.send_error(404, "Image not found")
            else:
                self.send_error(400, "Image name required")
        elif parsed.path.endswith("/api/thumb"):
            self.serve_thumb(task_path, params.get('name', [None])[0])
        elif parsed.path.endswith("/api/capabilities"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"pipeline": sc.pipeline_available()}).encode())
        elif parsed.path.endswith("/api/config"):
            config_path = params.get('config', [None])[0] or "config.toml"
            p = Path(config_path)
            if not p.is_absolute():
                p = self.root_dir / p
            allowed = [self.root_dir, task_path]
            safe = sc.resolve_within(str(p), allowed)

            if safe and safe.is_file():
                try:
                    json_str = sc.read_config_as_json(str(safe))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json_str.encode("utf-8"))
                except Exception as e:
                    self.send_error(500, str(e))
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        token = params.get('t', [None])[0]

        # Global SFX dictionary save — task-independent, so it runs before the token gate.
        if parsed.path.endswith("/api/sfx-dict/save"):
            self.save_sfx_dict()
            return

        if not token:
            self.send_error(403, "Token required")
            return

        self.token_manager.refresh()
        task_path = self.token_manager.get_task_path(token)
        if not task_path:
            self.send_error(404, "Invalid token or task not found")
            return

        if parsed.path.endswith("/api/collab/action"):
            user = params.get('user', [None])[0] or "Anonymous"
            task_name = self.token_manager.tokens.get(token)
            if not task_name:
                self.send_error(404, "Task not found")
                return
            self.handle_collab_action(params, task_name, user)
            return

        if parsed.path.endswith("/api/collab/chat"):
            user = params.get('user', [None])[0] or "Anonymous"
            task_name = self.token_manager.tokens.get(token)
            if not task_name:
                self.send_error(404, "Task not found")
                return
            self.handle_collab_chat(task_name, user)
            return

        if parsed.path.endswith("/api/save"):
            lang = params.get('lang', [None])[0]
            if lang and not sc.is_safe_lang(lang):
                self.send_error(400, "Invalid language format")
                return
            try:
                content_length = int(self.headers.get('Content-Length', 0) or 0)
                if content_length > 64 * 1024 * 1024:
                    raise ValueError("Payload too large")
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                # Validate at the write boundary so malformed/hostile payloads never
                # reach disk (and can't smuggle a traversal path into pages.json).
                try:
                    if lang:
                        sc.validate_translations_payload(data)
                    else:
                        sc.validate_pages_payload(data)
                except sc.PayloadError as e:
                    self.send_error(400, f"Invalid payload: {e}")
                    return
                if lang:
                    trans_dir = task_path / "translations"
                    trans_dir.mkdir(exist_ok=True)
                    target_file = trans_dir / f"{lang}.json"
                else:
                    target_file = task_path / "pages.json"

                with _SAVE_LOCK:
                    with open(target_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                mtime = 0
                if lang:
                    task_name = self.token_manager.tokens.get(token)
                    mtime = os.path.getmtime(target_file)
                    if task_name:
                        broadcast_save(task_name, lang, mtime)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "mtime": mtime}).encode())
            except Exception as e:
                self.send_error(500, str(e))
        elif parsed.path.endswith("/api/story/save"):
            try:
                content_length = int(self.headers.get('Content-Length', 0) or 0)
                if content_length > 64 * 1024 * 1024:
                    raise ValueError("Payload too large")
                post_data = self.rfile.read(content_length)
                target_file = task_path / "story.txt"
                with _SAVE_LOCK:
                    with open(target_file, "wb") as f:
                        f.write(post_data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
            except Exception as e:
                self.send_error(500, str(e))
        elif parsed.path.endswith("/api/page/delete"):
            try:
                content_length = int(self.headers.get('Content-Length', 0) or 0)
                if content_length > 64 * 1024 * 1024:
                    raise ValueError("Payload too large")
                post_data = self.rfile.read(content_length) if content_length else b'{}'
                payload = json.loads(post_data or b'{}')
                page_index = payload.get('page_index')
                clean_name = payload.get('clean')
                block_ids = set(payload.get('block_ids', []))
                result = {"status": "success", "image_removed": False, "langs_updated": [], "page_removed": False}

                with _SAVE_LOCK:
                    if page_index is not None:
                        pages_path = task_path / "pages.json"
                        with open(pages_path, "r", encoding="utf-8") as fh:
                            pages_data = json.load(fh)
                        pages = pages_data.get("pages") if isinstance(pages_data, dict) else None
                        if not isinstance(pages, list) or not isinstance(page_index, int) or page_index < 0 or page_index >= len(pages):
                            self.send_error(400, "Invalid page index")
                            return
                        pages.pop(page_index)

                    if clean_name and "/" not in clean_name and "\\" not in clean_name:
                        for sub in ("clean", "clean_v2"):
                            img_path = task_path / sub / clean_name
                            if img_path.exists():
                                img_path.unlink()
                                result["image_removed"] = True

                    trans_dir = task_path / "translations"
                    if block_ids and trans_dir.is_dir():
                        for f in trans_dir.iterdir():
                            if f.suffix != ".json" or not f.is_file():
                                continue
                            try:
                                with open(f, "r", encoding="utf-8") as fh:
                                    data = json.load(fh)
                            except Exception:
                                continue
                            if not isinstance(data, dict):
                                continue
                            removed = [bid for bid in block_ids if bid in data]
                            if removed:
                                for bid in removed:
                                    data.pop(bid, None)
                                with open(f, "w", encoding="utf-8") as fh:
                                    json.dump(data, fh, ensure_ascii=False, indent=2)
                                result["langs_updated"].append(f.stem)

                    if page_index is not None:
                        with open(task_path / "pages.json", "w", encoding="utf-8") as fh:
                            json.dump(pages_data, fh, ensure_ascii=False, indent=2)
                        result["page_removed"] = True

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_error(500, str(e))
        elif parsed.path.endswith("/api/image/inpaint-region"):
            try:
                content_length = int(self.headers.get('Content-Length', 0) or 0)
                if content_length > 1024 * 1024:
                    raise ValueError("Payload too large")
                payload = json.loads(self.rfile.read(content_length) or b'{}')
                self.inpaint_image_region(task_path, payload)
            except Exception as e:
                self.send_error(500, str(e))
        elif parsed.path.endswith("/api/image/inpaint-undo"):
            try:
                content_length = int(self.headers.get('Content-Length', 0) or 0)
                if content_length > 1024 * 1024:
                    raise ValueError("Payload too large")
                payload = json.loads(self.rfile.read(content_length) or b'{}')
                self.undo_inpaint_image_region(task_path, payload)
            except Exception as e:
                self.send_error(500, str(e))
        elif parsed.path.endswith("/api/pipeline/run"):
            try:
                content_length = int(self.headers.get('Content-Length', 0) or 0)
                if content_length > 64 * 1024 * 1024:
                    raise ValueError("Payload too large")
                post_data = self.rfile.read(content_length)
                args = json.loads(post_data)
                self.run_pipeline_task(args, task_path)
            except Exception as e:
                if not self.wfile.closed:
                    self.send_error(500, str(e))
        else:
            self.send_error(404)

    def inpaint_image_region(self, task_path, payload):
        img_name = payload.get("name")
        rect = payload.get("rect") or {}
        img_path = self.find_editable_image(task_path, img_name)
        if not img_path:
            return
        ext = os.path.splitext(img_name)[1].lower()

        try:
            from PIL import Image
            import numpy as np

            img = Image.open(img_path)
            image = np.array(img.convert("RGB"))
            h, w = image.shape[:2]
            x = max(0, min(w - 1, int(round(float(rect.get("x", 0))))))
            y = max(0, min(h - 1, int(round(float(rect.get("y", 0))))))
            rw = max(1, int(round(float(rect.get("w", 0)))))
            rh = max(1, int(round(float(rect.get("h", 0)))))
            x2 = max(x + 1, min(w, x + rw))
            y2 = max(y + 1, min(h, y + rh))
            if x2 - x < 4 or y2 - y < 4:
                self.send_error(400, "Region too small")
                return

            pad = max(2, int(round(float(payload.get("pad", 6)))))
            x = max(0, x - pad); y = max(0, y - pad)
            x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y:y2, x:x2] = 255

            from manga_translator_lite.config import Config, Inpainter
            from manga_translator_lite.inpainting import dispatch as dispatch_inpainting

            cfg = Config.load(None)
            cfg.inpainter.inpainter = Inpainter.lama_large
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
            except Exception:
                pass

            async def run():
                return await dispatch_inpainting(
                    cfg.inpainter.inpainter, image, mask, cfg.inpainter,
                    cfg.inpainter.inpainting_size, device=device)

            with EditorHandler.pipeline_lock:
                loop = asyncio.new_event_loop()
                try:
                    undo_dir = task_path / ".inpaint_undo"
                    undo_dir.mkdir(exist_ok=True)
                    shutil.copy2(img_path, undo_dir / f"{img_path.parent.name}__{img_name}")
                    result = loop.run_until_complete(run())
                    out = Image.fromarray(result)
                    fmt = img.format or ext.lstrip(".").upper()
                    save_kwargs = {}
                    if fmt == "JPEG" or ext in (".jpg", ".jpeg"):
                        fmt = "JPEG"; save_kwargs = {"quality": 95, "subsampling": 0, "optimize": True}
                    elif fmt == "WEBP" or ext == ".webp":
                        fmt = "WEBP"; save_kwargs = {"quality": 85, "method": 6}
                    elif fmt == "PNG" or ext == ".png":
                        fmt = "PNG"; save_kwargs = {"optimize": True}
                    out.save(img_path, format=fmt, **save_kwargs)
                finally:
                    loop.close()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "mtime": os.path.getmtime(img_path)}).encode())
        except ImportError as e:
            self.send_error(500, f"Inpainting unavailable: {e}")

    def undo_inpaint_image_region(self, task_path, payload):
        img_name = payload.get("name")
        img_path = self.find_editable_image(task_path, img_name)
        if not img_path:
            return
        undo_path = task_path / ".inpaint_undo" / f"{img_path.parent.name}__{img_name}"
        if not undo_path.exists():
            self.send_error(404, "No redraw undo available")
            return
        with EditorHandler.pipeline_lock:
            shutil.copy2(undo_path, img_path)
            undo_path.unlink()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "success", "mtime": os.path.getmtime(img_path)}).encode())

    def find_editable_image(self, task_path, img_name):
        if not img_name:
            self.send_error(400, "Image name required")
            return None
        ext = os.path.splitext(img_name)[1].lower()
        if "/" in img_name or "\\" in img_name or ext not in {".png", ".jpg", ".jpeg", ".webp"}:
            self.send_error(403, "Invalid image name")
            return None
        for sub in ("clean", "clean_v2"):
            p = task_path / sub / img_name
            if p.exists():
                return p
        self.send_error(404, "Image not found")
        return None

    def run_pipeline_task(self, args, task_path):
        cmd = args.get('cmd')
        target_lang = args.get('target_lang')
        overwrite = args.get('overwrite', False)
        start_index = args.get('start_index')
        input_path = args.get('input')
        output_path = args.get('output')
        config_path = args.get('config')
        reference_langs = sc.parse_reference_langs(args.get('reference_langs', 'auto'))

        try:
            input_path, output_path, config_path = sc.resolve_pipeline_paths(
                input_path, output_path, config_path, task_path, self.root_dir)
        except sc.PipelinePathError as e:
            self.send_error(400, str(e))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        log_q = queue.Queue()
        handler = LogQueueHandler(log_q)

        # Helper to send log
        def send_log(msg, type='log'):
            try:
                self.wfile.write(f"data: {json.dumps({'type': type, 'msg': msg})}\n\n".encode())
                self.wfile.flush()
            except:
                pass

        # Only one pipeline at a time; a concurrent request waits here so the
        # root-logger handler below never overlaps with another run.
        if not EditorHandler.pipeline_lock.acquire(blocking=False):
            send_log("Another pipeline task is running, waiting for it to finish...", 'status')
            EditorHandler.pipeline_lock.acquire()
        pipeline_logger = logging.getLogger('manga-translator')
        pipeline_logger.addHandler(handler)

        async def execute():
            try:
                await sc.run_pipeline(
                    cmd=cmd, task_path=task_path, config_path=config_path,
                    target_lang=target_lang, overwrite=overwrite, start_index=start_index,
                    reference_langs=reference_langs, input_path=input_path, output_path=output_path,
                    log=lambda kind, msg: send_log(msg, kind))
            finally:
                log_q.put(None)

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(execute())
            loop.close()

        threading.Thread(target=run_async).start()

        try:
            while True:
                msg = log_q.get()
                if msg is None: break
                send_log(msg)
        finally:
            pipeline_logger = logging.getLogger('manga-translator')
            pipeline_logger.removeHandler(handler)
            EditorHandler.pipeline_lock.release()

    # Largest dimension (px) of generated thumbnails for the editor's page list.
    THUMB_BOX = 160

    def serve_thumb(self, task_path, img_name):
        """Serve a small, disk-cached JPEG thumbnail of a clean image.

        Falls back to the original PNG if the name is missing/unsafe, and — crucially
        — if Pillow isn't installed or decoding fails, so the editor keeps working
        (just without the bandwidth savings) on a bare install.
        """
        if not img_name:
            self.send_error(400, "Image name required")
            return
        if "/" in img_name or "\\" in img_name:
            self.send_error(403, "Invalid image name")
            return
        src = None
        for sub in ("clean", "clean_v2"):
            p = task_path / sub / img_name
            if p.exists():
                src = p
                break
        if not src:
            self.send_error(404, "Image not found")
            return
        cache = sc.build_thumbnail(src, task_path / ".thumb_cache", self.THUMB_BOX, logger=self.logger)
        if cache:
            self.serve_file(cache, "image/jpeg")
        else:
            # Fallback to the original: label it by its real format (e.g. webp),
            # not a hardcoded PNG type.
            ct = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}
            self.serve_file(src, ct.get(os.path.splitext(img_name)[1].lower(), "image/png"))

    def handle_collab_stream(self, task_name, user):
        q = queue.Queue()
        with _COLLAB_LOCK:
            if task_name not in _LISTENERS:
                _LISTENERS[task_name] = []
            _LISTENERS[task_name].append(q)

            if task_name not in _ACTIVE_USERS:
                _ACTIVE_USERS[task_name] = {}
            _ACTIVE_USERS[task_name][user] = {
                "name": user,
                "avatar": "",
                "page": 0,
                "block": None,
                "lang": None,
                "ts": time.time()
            }

        broadcast_presence(task_name)

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        # Send in-memory chat history on connect
        with _COLLAB_LOCK:
            history = list(_CHAT_HISTORIES.get(task_name, []))
        for msg in history:
            self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode('utf-8'))
        self.wfile.flush()

        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode('utf-8'))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except Exception:
            pass
        finally:
            with _COLLAB_LOCK:
                if task_name in _LISTENERS and q in _LISTENERS[task_name]:
                    _LISTENERS[task_name].remove(q)
                if task_name in _ACTIVE_USERS and user in _ACTIVE_USERS[task_name]:
                    del _ACTIVE_USERS[task_name][user]
            broadcast_presence(task_name)

    def handle_collab_action(self, params, task_name, user):
        try:
            length = int(self.headers.get('Content-Length', 0) or 0)
            if length > 64 * 1024 * 1024: raise ValueError("Payload too large")
            action = json.loads(self.rfile.read(length))

            page = action.get("page", 0)
            block = action.get("block")
            lang = action.get("lang")

            with _COLLAB_LOCK:
                if task_name in _ACTIVE_USERS and user in _ACTIVE_USERS[task_name]:
                    _ACTIVE_USERS[task_name][user]["page"] = page
                    _ACTIVE_USERS[task_name][user]["block"] = block
                    _ACTIVE_USERS[task_name][user]["lang"] = lang
                    _ACTIVE_USERS[task_name][user]["ts"] = time.time()

            broadcast_presence(task_name)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode())
        except Exception as e:
            self.send_error(500, str(e))

    def handle_collab_chat(self, task_name, user):
        try:
            length = int(self.headers.get('Content-Length', 0) or 0)
            if length > 64 * 1024 * 1024: raise ValueError("Payload too large")
            action = json.loads(self.rfile.read(length))
            text = action.get("text", "")

            event = {
                "type": "chat",
                "user": user,
                "text": text,
                "ts": time.time()
            }

            with _COLLAB_LOCK:
                if task_name not in _CHAT_HISTORIES:
                    _CHAT_HISTORIES[task_name] = []
                _CHAT_HISTORIES[task_name].append(event)
                if len(_CHAT_HISTORIES[task_name]) > 30:
                    _CHAT_HISTORIES[task_name].pop(0)

            broadcast_event(task_name, event)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode())
        except Exception as e:
            self.send_error(500, str(e))

    # Global onomatopoeia / SFX candidate dictionary, maintained entirely by the
    # editor frontend and persisted next to editor.html (git-ignored). One shared file
    # across all tasks; the editor loads it at startup and re-saves on every change.
    SFX_DICT_NAME = "sfx_dictionary.json"

    def serve_sfx_dict(self):
        p = self.root_dir / self.SFX_DICT_NAME
        if p.exists():
            self.serve_file(p, "application/json")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    def save_sfx_dict(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0) or 0)
            if content_length > 8 * 1024 * 1024:
                raise ValueError("Payload too large")
            post_data = self.rfile.read(content_length) if content_length else b'{}'
            data = json.loads(post_data or b'{}')   # reject non-JSON before writing
            with _SAVE_LOCK:
                with open(self.root_dir / self.SFX_DICT_NAME, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode())
        except Exception as e:
            self.send_error(400, str(e))

    def serve_editor_html(self):
        """Serve editor.html, injecting a custom tool name (MTL_APP_NAME) if set."""
        path = self.root_dir / "editor.html"
        if not APP_NAME:
            self.serve_file(path, "text/html")
            return
        try:
            text = path.read_text(encoding="utf-8").replace(
                _EDITOR_NAME_MARKER, f"window.MTL_APP_NAME = {json.dumps(APP_NAME)};", 1)
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            self.serve_file(path, "text/html")

    def serve_file(self, path, content_type):
        if not path.exists():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", path.stat().st_size)
        self.end_headers()
        with open(path, "rb") as f:
            self.wfile.write(f.read())

    def log_message(self, format, *args):
        if self.logger:
            self.logger.info(format % args)

def run_server(work_dir, port=8000, host="127.0.0.1", log_file="server.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("server")

    secret_file = Path(work_dir) / ".secret"
    secret = None
    if secret_file.exists():
        try:
            secret = secret_file.read_text(encoding='utf-8').strip()
        except Exception:
            pass
    if not secret:
        secret = secrets.token_hex(16)
        try:
            Path(work_dir).mkdir(exist_ok=True)
            secret_file.write_text(secret, encoding='utf-8')
        except Exception:
            pass

    tm = TokenManager(work_dir, secret=secret)
    EditorHandler.token_manager = tm
    EditorHandler.root_dir = Path(__file__).parent.absolute()
    EditorHandler.logger = logger

    logger.info("="*60)
    logger.info(" Manga Translator Lite Standalone Server")
    logger.info(f" Port: {port}")
    logger.info(f" Work Directory: {os.path.abspath(work_dir)}")
    logger.info("="*60)

    links = tm.get_all_links(f"http://{host if host not in ('0.0.0.0', '127.0.0.1') else 'localhost'}:{port}/")
    if not links:
        logger.warning(" [!] No tasks found in work directory.")
    else:
        logger.info(" Available Task Links:")
        for name, link in links:
            logger.info(f" - {name}: {link}")
    base = f"http://{host if host not in ('0.0.0.0', '127.0.0.1') else 'localhost'}:{port}/"
    logger.info(f" Super Mode: {base}?super={tm.super_token}")

    with socketserver.ThreadingTCPServer((host, port), EditorHandler) as httpd:
        try:
            logger.info(f"Server started on http://{host}:{port}")
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--work-dir", default="work", help="Path to work directory")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--log-file", default="server.log", help="Path to log file")
    args = parser.parse_args()
    run_server(args.work_dir, args.port, args.host, args.log_file)
