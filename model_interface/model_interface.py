from abc import ABC, abstractmethod
from typing import Any, List


class ModelInterface(ABC):
    """Абстрактный базовый класс для всех моделей.

    Этот класс определяет интерфейс, который должны реализовывать все модели.
    Он включает методы для предсказания на основе одного или нескольких изображений.

    Attributes:
        model_name (str): Название модели как у разработчика (например, "Qwen2-VL-2B").
        system_prompt (str): Системный промпт, используемый моделью.
        cache_dir (str): Директория для кэширования данных модели.
        framework (str): Используемый фреймворк для инференса модели 
                        (например, "Hugging Face", "vLLM", "SGLang").
                        Каждый унаследованный класс реализует объект модели
                        привязанный к одному фреймворку инференса.
                        Этот атрибут класса всегда задан разработчиком класса для модели.
                        Атрибут не должен изменяться в процессе работы модели.
    """

    def __init__(self, model_name: str, system_prompt: str, cache_dir: str) -> None:
        """Инициализирует экземпляр Фабрики моделей.

        Args:
            model_name (str): Название модели как у разработчика (например, "Qwen2-VL-2B").
            system_prompt (str): Системный промпт, используемый моделью.
            cache_dir (str): Директория для кэширования данных модели.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir
        self.framework = ""

    @abstractmethod
    def predict_on_image(self, image: Any, question: str) -> str:
        """Абстрактный метод для предсказания на основе одного изображения.

        Args:
            image (Any): Изображение, на основе которого делается предсказание.
                        Тип может быть специфичным для реализации (например, PIL.Image, np.array и т.д.).
            question (str): промпт-вопрос по изображению изображению.

        Returns:
            str: строка с ответом от модели.
        """
        pass

    @abstractmethod
    def predict_on_images(self, images: List[Any], question: str) -> str:
        """Абстрактный метод для предсказания на основе нескольких изображений.

        Args:
            images (List[Any]): Список изображений, на основе которых делается предсказание.
                               Тип элементов списка может быть специфичным для реализации.
            question (str): промпт-вопрос по изображению изображению.

        Returns:
            str: строка с ответом от модели.
        """
        pass
