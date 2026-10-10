"""Opt-in workaround for a broken torchvision install in a text-only environment.

Put this directory first on PYTHONPATH only when Transformers discovers an
installed torchvision whose operators do not match the platform's PyTorch.
This does not modify or uninstall the platform's package.
"""

# ruff: noqa: INP001 -- PYTHONPATH 前缀钩子，本身没有包结构

import importlib.util

_find_spec = importlib.util.find_spec


def find_spec(name, *args, **kwargs):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _find_spec(name, *args, **kwargs)


importlib.util.find_spec = find_spec
