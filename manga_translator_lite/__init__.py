import colorama
from dotenv import load_dotenv

colorama.init(autoreset=True)
load_dotenv()

from .config import Config
from .utils import Context

__all__ = ["Config", "Context"]
