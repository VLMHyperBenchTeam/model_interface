import time
from typing import Callable, Any


def measure_inference_time(model_callable: Callable[..., Any], *args, **kwargs) -> float:
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


