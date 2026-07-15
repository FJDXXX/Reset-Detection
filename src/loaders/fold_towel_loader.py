from __future__ import annotations

from pathlib import Path

from src.loader import Episode
from .base_loader import BaseLoader


class FoldTowelLoader(BaseLoader):
    def load_episode(self, episode_path: Path) -> Episode:
        raise NotImplementedError("FoldTowelLoader not yet implemented")
