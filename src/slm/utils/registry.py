"""A tiny name->object registry (detectron2-style, trimmed).

Lets models / encoders / datasets register themselves under a string name so they can be
built from a config value. Usage:

    FOO_REGISTRY = Registry("FOO")

    @FOO_REGISTRY.register()
    class MyFoo: ...

    obj_cls = FOO_REGISTRY.get("MyFoo")
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        if name in self._obj_map:
            raise KeyError(f"'{name}' already registered in '{self._name}' registry")
        self._obj_map[name] = obj

    def register(self, obj: object = None, *, name: Optional[str] = None):
        """Use as a decorator (`@reg.register()`) or a call (`reg.register(obj)`)."""
        if obj is None:  # decorator form, possibly with an explicit name=
            def deco(func_or_class):
                self._do_register(name or func_or_class.__name__, func_or_class)
                return func_or_class
            return deco
        self._do_register(name or obj.__name__, obj)  # plain call form
        return obj

    def get(self, name: str):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' in '{self._name}' registry. "
                f"Registered: {sorted(self._obj_map)}"
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __iter__(self) -> Iterable[str]:
        return iter(self._obj_map)

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={sorted(self._obj_map)})"
