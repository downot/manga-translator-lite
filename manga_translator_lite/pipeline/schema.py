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
from typing import List, Optional, Tuple


WORKSPACE_VERSION = 2
PAGES_JSON = "pages.json"
CLEAN_DIR = "clean"


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
    translation: str = ""

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
            translation=str(data.get("translation", "")),
        )


@dataclass
class Page:
    index: int
    name: str                             # original filename (basename)
    size: Tuple[int, int]                 # (width, height)
    original: str                         # original input path (relative or absolute)
    clean: str                            # path to text-removed image, relative to workspace root
    blocks: List[Block] = field(default_factory=list)
    no_text: bool = False                 # True if no text was detected (OCR-empty page)

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
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Page":
        return cls(
            index=int(data["index"]),
            name=str(data.get("name", "")),
            size=tuple(data.get("size", [0, 0])),
            original=str(data.get("original", "")),
            clean=str(data.get("clean", "")),
            blocks=[Block.from_dict(b) for b in data.get("blocks", [])],
            no_text=bool(data.get("no_text", False)),
        )


@dataclass
class Workspace:
    root: str                             # absolute path to task workspace dir (the subdirectory)
    source_lang: str = "auto"
    target_lang: str = "ENG"
    task_name: str = ""                   # subdirectory name (task identifier)
    pages: List[Page] = field(default_factory=list)
    version: int = WORKSPACE_VERSION

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
        return d

    def all_blocks(self) -> List[Block]:
        out: List[Block] = []
        for p in self.pages:
            out.extend(p.blocks)
        return out

    def block_by_id(self, bid: str) -> Optional[Block]:
        for b in self.all_blocks():
            if b.id == bid:
                return b
        return None


def save_workspace(ws: Workspace) -> str:
    os.makedirs(ws.root, exist_ok=True)
    with open(ws.pages_json_path, 'w', encoding='utf-8') as f:
        json.dump(ws.to_dict(), f, ensure_ascii=False, indent=2)
    return ws.pages_json_path


def load_workspace(root: str) -> Workspace:
    root = os.path.abspath(root)
    path = os.path.join(root, PAGES_JSON)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Workspace metadata not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Workspace(
        root=root,
        version=int(data.get("version", WORKSPACE_VERSION)),
        source_lang=str(data.get("source_lang", "auto")),
        target_lang=str(data.get("target_lang", "ENG")),
        task_name=str(data.get("task_name", "")),
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
