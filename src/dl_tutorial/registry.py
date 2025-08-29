from typing import Any, Callable, Dict


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._store: Dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            key = name or getattr(obj, "__name__", None)
            if key is None:
                raise ValueError(f"Cannot register unnamed object in registry {self.name}")
            if key in self._store:
                raise KeyError(f"{key} already registered in {self.name}")
            self._store[key] = obj
            return obj
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._store:
            raise KeyError(f"{name} is not registered in {self.name}. Available: {list(self._store)}")
        return self._store[name]

    def __contains__(self, name: str) -> bool:  # pragma: no cover
        return name in self._store


BACKBONES = Registry("backbones")
HEADS = Registry("heads")
LOSSES = Registry("losses")
MODELS = Registry("models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
