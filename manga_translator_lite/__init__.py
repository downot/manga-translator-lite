import os

# Reduce CUDA memory fragmentation from the detect(2560)/inpaint(2048) size alternation
# in extract. This MUST be set before torch initializes its CUDA caching allocator, so it
# lives at the very top of the package init (which runs before any submodule imports torch).
# setdefault → a value the user exported in the environment always wins. On platforms /
# torch builds that don't support `expandable_segments`, torch just ignores it (harmless).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import colorama
from dotenv import load_dotenv

colorama.init(autoreset=True)
load_dotenv()

from .config import Config
from .utils import Context

__all__ = ["Config", "Context"]
