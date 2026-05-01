from dataclasses import dataclass, field, asdict
from typing import List, Dict

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
