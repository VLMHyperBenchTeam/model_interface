"""Model Interface Package

Пакет предоставляет интерфейс для работы с различными моделями VLM.
"""

__version__ = "0.1.2.dev1"

from .model_interface import ModelInterface
from .model_factory import ModelFactory, load_prompt
from .model_utils import measure_inference_time

__all__ = [
    "ModelInterface",
    "ModelFactory", 
    "load_prompt",
    "measure_inference_time",
    "__version__"
]
