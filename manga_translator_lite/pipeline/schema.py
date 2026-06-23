"""On-disk schema for the intermediate workspace.

A workspace directory looks like this (subdirectory-based multi-task layout):

    work_dir/
        task_a/
            pages.json         # Workspace for task_a
            clean/0001.png     # Inpainted (text-removed) images
            clean/0002.png
        task_b/
            pages.json
            clean/...

Each subdirectory under work_dir is an independent task workspace.
`pages.json` is the single source of truth per task. The translate step writes
translation strings back into the same file. Users may edit pages.json
between translate and render to revise translations.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


WORKSPACE_VERSION = 4
PAGES_JSON = "pages.json"
CLEAN_DIR = "clean"
TRANSLATIONS_DIR = "translations"


@dataclass
class Translation:
    text: str = ""
    edited: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Translation":
        return cls(
            text=str(data.get("text", "")),
            edited=bool(data.get("edited", False))
        )


def block_id(page_idx: int, block_idx: int) -> str:
    return f"p{page_idx:04d}_b{block_idx:03d}"


@dataclass
class Block:
    id: str
    text: str
    bbox: List[int]                       # [x, y, w, h]
    polygon: List[List[int]]              # 4-point polygon, ints
    lines: List[List[List[int]]]          # list of 4-point polygons (per textline)
    ocr_text: str = ""                     # original OCR result (never edited)
    font_size: int = 0
    angle: float = 0.0
    fg_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    bg_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    direction: str = "auto"               # auto | h | v | hr | vr
    alignment: str = "auto"               # auto | left | center | right
    prob: float = 1.0
    bg_fill: str = "none"                  # "none" (transparent) | "match" (fill bg_color) | "white" (fill white)
    user_added: bool = False               # manually drawn in the editor; preserved across re-extract
    fixed_region: bool = False             # box is user-controlled → render fits text into it, never auto-expands
    scale_exempt: bool = False             # reserved: when True this block ignores the task-level box_scale

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Block":
        return cls(
            id=data["id"],
            text=data.get("text", ""),
            ocr_text=data.get("ocr_text", data.get("text", "")),
            bbox=list(data.get("bbox", [0, 0, 0, 0])),
            polygon=[list(p) for p in data.get("polygon", [])],
            lines=[[list(p) for p in line] for line in data.get("lines", [])],
            font_size=int(data.get("font_size", 0)),
            angle=float(data.get("angle", 0.0)),
            fg_color=list(data.get("fg_color", [0, 0, 0])),
            bg_color=list(data.get("bg_color", [255, 255, 255])),
            direction=str(data.get("direction", "auto")),
            alignment=str(data.get("alignment", "auto")),
            prob=float(data.get("prob", 1.0)),
            bg_fill=str(data.get("bg_fill", "none")),
            user_added=bool(data.get("user_added", False)),
            fixed_region=bool(data.get("fixed_region", False)),
            scale_exempt=bool(data.get("scale_exempt", False)),
        )


@dataclass
class Page:
    index: int
    name: str                             # original filename (basename)
    size: Tuple[int, int]                 # (width, height)
    original: str                         # original filename (basename) for identification
    clean: str                            # path to text-removed image, relative to workspace root
    blocks: List[Block] = field(default_factory=list)
    no_text: bool = False                 # True if no text was detected (OCR-empty page)
    # Regions detected as text but rejected by the translation rules (empty OCR,
    # symbols, handwritten kana, too-short). They are still erased during extract;
    # stored here as 4-point polygons so reclean can rebuild the full erase mask.
    erase_regions: List[List[List[int]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "index": self.index,
            "name": self.name,
            "size": list(self.size),
            "original": self.original,
            "clean": self.clean,
            "blocks": [b.to_dict() for b in self.blocks],
        }
        if self.no_text:
            d["no_text"] = True
        if self.erase_regions:
            d["erase_regions"] = self.erase_regions
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Page":
        return cls(
            index=int(data["index"]),
            # name/original feed the render OUTPUT filename, so reduce them to a bare
            # basename: a client-written pages.json must never steer writes with a path
            # like "../../x.png" or "/etc/x". (page.clean is a workspace-relative READ
            # path and is confined against the root at use; see safe_workspace_path.)
            name=os.path.basename(str(data.get("name", ""))),
            size=tuple(data.get("size", [0, 0])),
            original=os.path.basename(str(data.get("original", "")).replace("\\", "/")),
            clean=str(data.get("clean", "")).replace("\\", "/"),
            blocks=[Block.from_dict(b) for b in data.get("blocks", [])],
            no_text=bool(data.get("no_text", False)),
            erase_regions=[[[int(p[0]), int(p[1])] for p in poly]
                           for poly in data.get("erase_regions", [])],
        )


@dataclass
class Workspace:
    root: str                             # absolute path to task workspace dir (the subdirectory)
    source_lang: str = "auto"
    target_lang: str = "ENG"
    task_name: str = ""                   # subdirectory name (task identifier)
    pages: List[Page] = field(default_factory=list)
    version: int = WORKSPACE_VERSION
    box_scale: float = 1.0                 # task-level uniform text-box enlargement applied at render/preview
    font_size_minimum: Optional[int] = None              # per-task override; None → fall back to config.toml
    font_size_minimum_expand_limit: Optional[float] = None  # per-task override; None → fall back to config.toml

    @property
    def pages_json_path(self) -> str:
        return os.path.join(self.root, PAGES_JSON)

    @property
    def clean_dir(self) -> str:
        return os.path.join(self.root, CLEAN_DIR)

    def to_dict(self) -> dict:
        d = {
            "version": self.version,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "pages": [p.to_dict() for p in self.pages],
        }
        if self.task_name:
            d["task_name"] = self.task_name
        if self.box_scale and self.box_scale != 1.0:
            d["box_scale"] = self.box_scale
        if self.font_size_minimum is not None:
            d["font_size_minimum"] = self.font_size_minimum
        if self.font_size_minimum_expand_limit is not None:
            d["font_size_minimum_expand_limit"] = self.font_size_minimum_expand_limit
        return d

    def all_blocks(self) -> List[Block]:
        out: List[Block] = []
        for p in self.pages:
            out.extend(p.blocks)
        return out

def save_workspace(ws: Workspace) -> str:
    os.makedirs(ws.root, exist_ok=True)
    with open(ws.pages_json_path, 'w', encoding='utf-8') as f:
        json.dump(ws.to_dict(), f, ensure_ascii=False, indent=2)
    return ws.pages_json_path


def get_translations_dir(root: str) -> str:
    return os.path.join(root, TRANSLATIONS_DIR)


def get_translation_path(root: str, lang: str) -> str:
    return os.path.join(get_translations_dir(root), f"{lang}.json")


def load_translations(root: str, lang: str) -> Dict[str, Translation]:
    path = get_translation_path(root, lang)
    if not os.path.exists(path):
        return {}
    if os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {bid: Translation.from_dict(t) for bid, t in data.items()}
    except json.JSONDecodeError:
        return {}


def save_translations(root: str, lang: str, translations: Dict[str, Translation]):
    os.makedirs(get_translations_dir(root), exist_ok=True)
    path = get_translation_path(root, lang)
    data = {bid: t.to_dict() for bid, t in translations.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_workspace_path(root: str, rel: str) -> Optional[str]:
    """Resolve a workspace-relative path under ``root``.

    Returns the absolute path only if it stays inside ``root``; returns None for an
    empty value or any path that escapes (``..`` segments, an absolute path, or a
    symlink-free traversal). Used to confine client-writable fields such as
    ``page.clean`` before they are opened/copied at render time, so a tampered
    pages.json can't read/write files outside the task workspace.
    """
    if not rel:
        return None
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, rel))
    if target == root_abs or target.startswith(root_abs + os.sep):
        return target
    return None


def load_workspace(root: str) -> Workspace:
    root = os.path.abspath(root)
    path = os.path.join(root, PAGES_JSON)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Workspace metadata not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Workspace metadata file is empty: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode workspace metadata {path}: {e}")
    return Workspace(
        root=root,
        version=int(data.get("version", WORKSPACE_VERSION)),
        source_lang=str(data.get("source_lang", "auto")),
        target_lang=str(data.get("target_lang", "ENG")),
        task_name=str(data.get("task_name", "")),
        box_scale=float(data.get("box_scale", 1.0)),
        font_size_minimum=(int(data["font_size_minimum"]) if data.get("font_size_minimum") is not None else None),
        font_size_minimum_expand_limit=(float(data["font_size_minimum_expand_limit"]) if data.get("font_size_minimum_expand_limit") is not None else None),
        pages=[Page.from_dict(p) for p in data.get("pages", [])],
    )


def discover_tasks(work_dir: str) -> List[str]:
    """Return a sorted list of task subdirectory names under work_dir.

    Only subdirectories that contain a ``pages.json`` file are considered
    valid task workspaces.
    """
    work_dir = os.path.abspath(work_dir)
    tasks = []
    if not os.path.isdir(work_dir):
        return tasks
    for entry in sorted(os.listdir(work_dir)):
        full = os.path.join(work_dir, entry)
        if os.path.isdir(full) and not entry.startswith('.'):
            if os.path.isfile(os.path.join(full, PAGES_JSON)):
                tasks.append(entry)
    return tasks
