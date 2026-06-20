import http.server
import socketserver
import json
import os
import secrets
import hashlib
import urllib.parse
import sys
import threading
import queue
import asyncio
import logging
from pathlib import Path

# Add project root to sys.path so manga_translator_lite can be imported *if present*.
sys.path.append(str(Path(__file__).parent.absolute()))

# Customizable tool name (tab title + header brand in editor.html). Empty = default.
APP_NAME = os.environ.get("MTL_APP_NAME", "").strip()
_EDITOR_NAME_MARKER = 'window.MTL_APP_NAME = "Manga Translator";'

# The editor itself (view/edit translations, pages, images, story) runs WITHOUT the
# manga_translator_lite package. Only the pipeline runner (/api/pipeline/run) needs it,
# so those imports are deferred to that endpoint — letting the editor start standalone.


def _pipeline_available() -> bool:
    """Whether the pipeline (extract/translate/render) can run at all.

    Cheap presence check: is the manga_translator_lite package importable *by
    location*? Uses find_spec so we DON'T import its heavy ML deps just to probe.
    The editor stays standalone; this only decides whether the editor shows the
    Pipeline tab as usable. The actual run still guards with try/except ImportError
    (so a package that's present but has missing deps degrades to a clear message).
    """
    try:
        import importlib.util
        return importlib.util.find_spec("manga_translator_lite") is not None
    except Exception:
        return False


def _read_config_as_json(path: str) -> str:
    """Return the config file as a JSON string for the editor's render preview.

    Prefers manga_translator_lite.Config (applies defaults/validation); falls back
    to a plain stdlib TOML/JSON parse so the editor works without the package.
    Never raises — returns "{}" if nothing can be read (editor uses its defaults).
    """
    try:
        from manga_translator_lite.config import Config
        cfg = Config.load(path)
        return cfg.model_dump_json() if hasattr(cfg, "model_dump_json") else cfg.json()
    except Exception:
        pass
    try:
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if ext == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            return json.dumps(tomllib.loads(content))
        return json.dumps(json.loads(content))
    except Exception:
        return "{}"


def _is_safe_lang(lang: str) -> bool:
    """Validate that language name only contains alphanumeric characters, hyphens or underscores.

    Prevents directory traversal attacks via query parameters.
    """
    import re
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", lang))


def _is_safe_config_path(path_str: str, root_dir: Path) -> bool:
    """Ensure that the resolved configuration path resides within the root directory.

    Prevents reading arbitrary configuration files on the system.
    """
    try:
        resolved = Path(path_str).resolve()
        base = root_dir.resolve()
        return base in resolved.parents or resolved == base
    except Exception:
        return False


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
        self.tokens = {} # token -> task_name
        self.tasks = {} # task_name -> token
        self.refresh()

    def refresh(self):
        if not self.work_dir.exists():
            return
        for item in self.work_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
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

_COLLAB_LOCK = threading.Lock()
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
                if not _is_safe_lang(lang):
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
                if "/" in img_name or "\\" in img_name:
                    self.send_error(403, "Invalid image name")
                    return
                # Check different possible image locations
                for sub in ["clean", "clean_v2"]:
                    p = task_path / sub / img_name
                    if p.exists():
                        self.serve_file(p, "image/png")
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
            self.wfile.write(json.dumps({"pipeline": _pipeline_available()}).encode())
        elif parsed.path.endswith("/api/config"):
            config_path = params.get('config', [None])[0]
            if not config_path:
                config_path = "config.toml"
            resolved_path = Path(config_path)
            if not resolved_path.is_absolute():
                resolved_path = self.root_dir / resolved_path
            
            if not _is_safe_config_path(str(resolved_path), self.root_dir):
                self.send_error(403, "Access to configuration file denied")
                return

            if resolved_path.exists() and resolved_path.is_file():
                try:
                    json_str = _read_config_as_json(str(resolved_path))
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
            if lang and not _is_safe_lang(lang):
                self.send_error(400, "Invalid language format")
                return
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                if lang:
                    trans_dir = task_path / "translations"
                    trans_dir.mkdir(exist_ok=True)
                    target_file = trans_dir / f"{lang}.json"
                else:
                    target_file = task_path / "pages.json"

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
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                target_file = task_path / "story.txt"
                with open(target_file, "wb") as f:
                    f.write(post_data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
            except Exception as e:
                self.send_error(500, str(e))
        elif parsed.path.endswith("/api/page/delete"):
            content_length = int(self.headers.get('Content-Length', 0) or 0)
            post_data = self.rfile.read(content_length) if content_length else b'{}'
            try:
                payload = json.loads(post_data or b'{}')
                clean_name = payload.get('clean')
                block_ids = set(payload.get('block_ids', []))
                result = {"status": "success", "image_removed": False, "langs_updated": []}

                # 1. Delete the clean image (guard against path traversal).
                if clean_name and "/" not in clean_name and "\\" not in clean_name:
                    for sub in ("clean", "clean_v2"):
                        img_path = task_path / sub / clean_name
                        if img_path.exists():
                            try:
                                img_path.unlink()
                                result["image_removed"] = True
                            except Exception as e:
                                if self.logger:
                                    self.logger.warning(f"Failed to delete image {img_path}: {e}")

                # 2. Strip the block ids from every language's translation file.
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

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_error(500, str(e))
        elif parsed.path.endswith("/api/pipeline/run"):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                args = json.loads(post_data)
                self.run_pipeline_task(args, task_path)
            except Exception as e:
                if not self.wfile.closed:
                    self.send_error(500, str(e))
        else:
            self.send_error(404)

    def run_pipeline_task(self, args, task_path):
        cmd = args.get('cmd')
        target_lang = args.get('target_lang')
        overwrite = args.get('overwrite', False)
        start_index = args.get('start_index')
        input_path = args.get('input')
        output_path = args.get('output')
        config_path = args.get('config')
        # Cross-language reference: 'auto' (default) → None (all reviewed langs),
        # 'off' → [] (no reference), or a list of codes → manual.
        _ref = args.get('reference_langs', 'auto')
        reference_langs = [] if _ref == 'off' else (_ref if isinstance(_ref, list) else None)

        if config_path:
            resolved_path = Path(config_path)
            if not resolved_path.is_absolute():
                resolved_path = self.root_dir / resolved_path
            if not _is_safe_config_path(str(resolved_path), self.root_dir):
                self.send_error(403, "Access to configuration file denied")
                return
            config_path = str(resolved_path)

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
            # Pipeline features need the full package — import lazily so the editor
            # itself can run without manga_translator_lite (and its ML deps) installed.
            try:
                from manga_translator_lite.config import Config
                from manga_translator_lite.pipeline.extract import run_extract
                from manga_translator_lite.pipeline.translate import run_translate
                from manga_translator_lite.pipeline.render import run_render
            except ImportError as e:
                send_log(f"Pipeline unavailable: manga_translator_lite is not importable ({e}). "
                         f"The editor runs standalone, but extract/translate/render need the full package.",
                         'error')
                log_q.put(None)
                return

            cfg = Config.load(config_path or None)
            if target_lang:
                cfg.translator.target_lang = target_lang

            try:
                if cmd == 'extract':
                    await run_extract(input_path or str(task_path / "in"), task_path, cfg, overwrite=overwrite)
                elif cmd == 'translate':
                    await run_translate(task_path, cfg, overwrite=overwrite, target_lang=target_lang,
                                        start_index=start_index, reference_langs=reference_langs)
                elif cmd == 'render':
                    await run_render(task_path, output_path or str(task_path / "out"), cfg)
                elif cmd == 'run':
                    await run_extract(input_path or str(task_path / "in"), task_path, cfg, overwrite=overwrite)
                    await run_translate(task_path, cfg, overwrite=overwrite, target_lang=target_lang,
                                        reference_langs=reference_langs)
                    await run_render(task_path, output_path or str(task_path / "out"), cfg)
                send_log("--- Pipeline Finished ---", 'status')
            except Exception as e:
                send_log(f"Error: {str(e)}", 'error')
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

        Falls back to the original PNG if the name is missing/unsafe is rejected,
        and — crucially — if Pillow isn't installed or decoding fails, so the editor
        keeps working (just without the bandwidth savings) on a bare install.
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
        try:
            from PIL import Image
            cache_dir = task_path / ".thumb_cache"
            cache_path = cache_dir / (img_name + ".jpg")
            # Regenerate when missing or older than the source image.
            if (not cache_path.exists()) or cache_path.stat().st_mtime < src.stat().st_mtime:
                cache_dir.mkdir(exist_ok=True)
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im.thumbnail((self.THUMB_BOX, self.THUMB_BOX))
                    # Write to a unique temp file then atomically rename, so a
                    # concurrent request never reads a half-written thumbnail.
                    import tempfile
                    fd, tmp = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
                    os.close(fd)
                    im.save(tmp, "JPEG", quality=72)
                    os.replace(tmp, cache_path)
            self.serve_file(cache_path, "image/jpeg")
        except Exception as e:
            # Pillow absent or decode error → serve the full image instead.
            if self.logger:
                self.logger.info(f"Thumbnail fallback for {img_name}: {e}")
            self.serve_file(src, "image/png")

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
            length = int(self.headers['Content-Length'])
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
            length = int(self.headers['Content-Length'])
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
