"""Dependency-free input discovery and source-change helpers for extract."""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def _natural_sort(names: List[str]) -> List[str]:
    return sorted(names, key=lambda text: [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', text)])


def list_images(input_path: str) -> List[str]:
    """Return only the images explicitly contained by a file or directory."""
    if os.path.isfile(input_path):
        return [input_path]
    return [
        os.path.join(input_path, name)
        for name in _natural_sort(os.listdir(input_path))
        if not name.startswith('.') and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]


def discover_input_tasks(input_path: str) -> List[Tuple[str, List[str]]]:
    """Group an input file or directory into named, non-empty image tasks.

    A single file remains a one-image task. When a directory mixes root images
    with image-bearing subdirectories, the root images become their own task
    instead of being silently discarded.
    """
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        parent = os.path.dirname(input_path)
        task_name = os.path.basename(parent) or os.path.splitext(os.path.basename(input_path))[0]
        return [(task_name, [input_path])]

    root_images = list_images(input_path)
    sub_tasks: List[Tuple[str, List[str]]] = []
    for entry in _natural_sort(os.listdir(input_path)):
        full = os.path.join(input_path, entry)
        if not entry.startswith('.') and os.path.isdir(full):
            images = list_images(full)
            if images:
                sub_tasks.append((entry, images))

    if sub_tasks:
        if root_images:
            root_name = os.path.basename(input_path.rstrip(os.sep)) or 'input'
            return [(root_name, root_images), *sub_tasks]
        return sub_tasks

    if root_images:
        task_name = os.path.basename(input_path.rstrip(os.sep)) or 'input'
        return [(task_name, root_images)]
    return []


def source_fingerprint(path: str) -> Tuple[int, int]:
    """Return stable-enough metadata for incremental extraction decisions."""
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


def source_matches(path: str, size: Optional[int], mtime_ns: Optional[int]) -> bool:
    """Return whether a stored source fingerprint still matches ``path``."""
    if size is None or mtime_ns is None:
        return False
    return source_fingerprint(path) == (size, mtime_ns)
