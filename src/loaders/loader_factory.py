from __future__ import annotations

from .base_loader import BaseLoader
from .yuanli_loader import YuanliLoader
from .fold_towel_loader import FoldTowelLoader
from .kuavo_loader import KuavoLoader


class LoaderFactory:
    _registry: dict[str, type[BaseLoader]] = {
        "yuanli": YuanliLoader,
        "fold_towel": FoldTowelLoader,
        "kuavo": KuavoLoader,
    }

    @classmethod
    def get_loader(cls, robot_name: str) -> BaseLoader:
        loader_cls = cls._registry.get(robot_name)
        if loader_cls is None:
            raise ValueError(f"Unknown robot: {robot_name}. Available: {list(cls._registry.keys())}")
        return loader_cls()

    @classmethod
    def register(cls, robot_name: str, loader_cls: type[BaseLoader]) -> None:
        cls._registry[robot_name] = loader_cls
