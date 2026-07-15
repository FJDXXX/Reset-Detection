from __future__ import annotations

from .base_loader import BaseLoader
from .yuanli_loader import YuanliLoader
from .told_towel_loader import ToldTowelLoader


class LoaderFactory:
    _registry: dict[str, type[BaseLoader]] = {
        "yuanli": YuanliLoader,
        "told_towel": ToldTowelLoader,
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
