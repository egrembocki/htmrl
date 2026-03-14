from importlib import import_module
from typing import TYPE_CHECKING, Any

HTM_MODULE = "psu_capstone.agent_layer.HTM"

if TYPE_CHECKING:
    from psu_capstone.agent_layer.agent import Agent
    from psu_capstone.agent_layer.agent_interface import AgentInterface
    from psu_capstone.agent_layer.brain import Brain
    from psu_capstone.agent_layer.HTM import ColumnField, Field, InputField, OutputField
    from psu_capstone.agent_layer.train import Trainer

__all__ = [
    "Agent",
    "AgentInterface",
    "Brain",
    "ColumnField",
    "Field",
    "InputField",
    "OutputField",
    "Trainer",
]

_EXPORTS = {
    "Agent": ("psu_capstone.agent_layer.agent", "Agent"),
    "AgentInterface": ("psu_capstone.agent_layer.agent_interface", "AgentInterface"),
    "Brain": ("psu_capstone.agent_layer.brain", "Brain"),
    "ColumnField": (HTM_MODULE, "ColumnField"),
    "Field": (HTM_MODULE, "Field"),
    "InputField": (HTM_MODULE, "InputField"),
    "OutputField": (HTM_MODULE, "OutputField"),
    "Trainer": ("psu_capstone.agent_layer.train", "Trainer"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
