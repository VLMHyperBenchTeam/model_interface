import importlib
from typing import Any, Dict, Optional, Type


class ModelFactory:
    """Фабрика для создания и управления зарегистрированными моделями."""

    # Словарь для хранения зарегистрированных моделей
    # {model_name: model_class}
    _models: Dict[str, Type[Any]] = {}

    @classmethod
    def register_model(cls, model_name: str, model_path: str) -> None:
        """Регистрирует класс модели по её пути в python-пакете.

        Args:
            model_name (str): Имя модели, под которым она будет зарегистрирована.
            model_path (str): Путь к классу модели в формате
                'python_package_name.module_with_class:class_name'.

        Raises:
            ImportError: Если модуль не может быть импортирован.
            AttributeError: Если класс не найден в указанном модуле.
        """
        module_path, class_name = model_path.split(":")
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        cls._models[model_name] = model_class

    @classmethod
    def get_model(
        cls, model_name: str, model_init_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Создает и возвращает экземпляр модели по её имени.

        Args:
            model_name (str): Имя зарегистрированной модели.
            model_init_params (Optional[Dict[str, Any]]): Параметры для инициализации модели.
                Если None, передается пустой словарь.

        Returns:
            Any: Экземпляр модели.

        Raises:
            ValueError: Если модель с указанным именем не зарегистрирована.
        """
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' is not registered.")
        if model_init_params is None:
            model_init_params = {}
        return cls._models[model_name](**model_init_params)
