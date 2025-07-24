import json
import time
from collections.abc import Callable
from typing import Any


def measure_inference_time(model_callable: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """
    Измеряет время выполнения (inference time) программы или функции.

    Args:
        model_callable: Функция или метод модели, который нужно измерить.
        *args: Аргументы для передачи в model_callable.
        **kwargs: Ключевые аргументы для передачи в model_callable.

    Returns:
        Время выполнения в секундах.

    Использование:
        model = MyModel()
        input_data = "Пример входных данных"
        inference_time = measure_inference_time(model.predict, input_data)
        print(f"Время выполнения: {inference_time:.4f} секунд")
    """

    start_time = time.time()
    model_callable(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time


def load_model_config(config_path: str) -> dict:
    """Загружает конфигурацию модели из JSON файла."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON в файле {config_path}: {e}") from e
