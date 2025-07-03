import importlib
from pathlib import Path
import threading
from typing import Any, Protocol

# Константы (универсальные)
DEFAULT_CACHE_DIR = "model_cache"
DEFAULT_DEVICE_MAP = "auto"


class ModelInterface(Protocol):
    """Протокол для определения интерфейса модели."""

    def predict_on_image(self, image: Any, prompt: str) -> str:
        """Предсказание на основе изображения/пути/URL и промпта."""
        ...


class ModelFactory:
    """Фабрика для создания моделей.

    Класс предоставляет методы для регистрации моделей и создания их экземпляров.
    Модели регистрируются по имени и пути к классу в Python-пакете.
    Класс потокобезопасен и поддерживает валидацию входных данных.

    Attributes:
        _models (Dict[str, Type[ModelInterface]]): Словарь для хранения зарегистрированных моделей,
                                                  где ключ — имя модели, а значение — класс модели.
        _lock (threading.Lock): Блокировка для потокобезопасности.
    """

    _models: dict[str, type[ModelInterface]] = {}
    _lock = threading.Lock()

    @classmethod
    def register_model(cls, model_name: str, model_path: str) -> None:
        """Регистрирует класс модели по её пути в python-пакете.

        Args:
            model_name (str): Имя модели, под которым она будет зарегистрирована.
            model_path (str): Путь к классу модели в формате 'module_path:class_name'.

        Raises:
            ValueError: Если неверный формат пути модели или пустые параметры.
            ImportError: Если модуль не может быть импортирован.
            AttributeError: Если класс не найден в указанном модуле.
        """
        # Валидация входных данных
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name должен быть непустой строкой")

        if not model_path or not isinstance(model_path, str):
            raise ValueError("model_path должен быть непустой строкой")

        if ":" not in model_path:
            raise ValueError(
                "model_path должен быть в формате 'module_path:class_name'"
            )

        try:
            module_path, class_name = model_path.split(":", 1)  # Разделяем только первый ':'

            if not module_path or not class_name:
                raise ValueError("Неверный формат пути модели")

            print(f"INFO: Попытка импорта модуля: {module_path}")
            module = importlib.import_module(module_path)

            if not hasattr(module, class_name):
                raise AttributeError(f"Класс '{class_name}' не найден в модуле '{module_path}'")

            model_class = getattr(module, class_name)

            # Потокобезопасная регистрация
            with cls._lock:
                cls._models[model_name] = model_class
                print(f"INFO: Модель '{model_name}' успешно зарегистрирована")

        except ImportError as e:
            print(f"ERROR: Ошибка импорта модуля '{module_path}': {str(e)}")
            raise ImportError(f"Не удалось импортировать модуль '{module_path}': {str(e)}") from e
        except AttributeError as e:
            print(f"ERROR: Ошибка получения класса '{class_name}': {str(e)}")
            raise AttributeError(f"Класс '{class_name}' не найден в модуле '{module_path}': {str(e)}") from e

    @classmethod
    def get_model(
        cls, model_name: str, model_init_params: dict[str, Any] | None = None
    ) -> ModelInterface:
        """Создает и возвращает экземпляр модели по её имени.

        Args:
            model_name (str): Имя зарегистрированной модели.
            model_init_params (Optional[Dict[str, Any]]): Параметры для инициализации модели.
                Если None, передается пустой словарь.

        Returns:
            ModelInterface: Экземпляр модели.

        Raises:
            ValueError: Если модель с указанным именем не зарегистрирована или неверные параметры.
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name должен быть непустой строкой")

        if model_init_params is None:
            model_init_params = {}
        elif not isinstance(model_init_params, dict):
            raise ValueError("model_init_params должен быть словарем")

        with cls._lock:
            if model_name not in cls._models:
                available_models = list(cls._models.keys())
                raise ValueError(
                    f"Модель '{model_name}' не зарегистрирована. "
                    f"Доступные модели: {available_models}"
                )

            model_class = cls._models[model_name]

        try:
            print(f"INFO: Создание экземпляра модели '{model_name}' с параметрами: {model_init_params}")
            return model_class(**model_init_params)
        except Exception as e:
            print(f"ERROR: Ошибка создания экземпляра модели '{model_name}': {str(e)}")
            raise ValueError(f"Ошибка создания экземпляра модели '{model_name}': {str(e)}") from e

    @classmethod
    def get_registered_models(cls) -> dict[str, type[ModelInterface]]:
        """Возвращает копию словаря зарегистрированных моделей.

        Returns:
            Dict[str, Type[ModelInterface]]: Копия словаря моделей.
        """
        with cls._lock:
            return cls._models.copy()

    @classmethod
    def is_model_registered(cls, model_name: str) -> bool:
        """Проверяет, зарегистрирована ли модель.

        Args:
            model_name (str): Имя модели для проверки.

        Returns:
            bool: True, если модель зарегистрирована, иначе False.
        """
        with cls._lock:
            return model_name in cls._models

    @classmethod
    def initialize_model(cls, model_config: dict[str, Any]) -> ModelInterface:
        """Инициализирует и возвращает модель согласно вложенной конфигурации.

        Ожидаем словарь следующего вида::

            {
                "common_params": {
                    "model_family": "Qwen2.5-VL",
                    "model_name": "Qwen2.5-VL-7B-Instruct",
                    "cache_dir": "/tmp",
                    "device_map": "auto",
                    "system_prompt": "...",           # опционально
                    "package": "...", "module": "...", "model_class": "..."  # при необходимости регистрации
                },
                "specific_params": {
                    "max_size": 2048 * 28 * 28,
                    ...                 # любые поля, понятные семейству модели
                }
            }

        Мы намеренно нарушаем обратную совместимость: прежний «плоский» формат
        больше не поддерживается.
        """

        if not isinstance(model_config, dict):
            raise ValueError("model_config должен быть словарём")

        if "common_params" not in model_config or not isinstance(model_config["common_params"], dict):
            raise KeyError("Отсутствует обязательный ключ 'common_params' или он не словарь")

        common_params: dict[str, Any] = model_config["common_params"]
        model_config.get("specific_params", {})

        # ----------------------------------------------
        # Валидация common_params
        # ----------------------------------------------
        required_keys = {"model_family", "model_name", "cache_dir", "device_map"}
        if missing := required_keys - common_params.keys():
            raise KeyError(f"В 'common_params' отсутствуют обязательные ключи: {missing}")

        for key in required_keys:
            if not common_params[key]:
                raise ValueError(f"Значение '{key}' в 'common_params' не может быть пустым")

        # ----------------------------------------------
        # Подготовка директории кэша
        # ----------------------------------------------
        cache_dir = Path(common_params["cache_dir"])
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"INFO: Директория кэша создана/проверена: {cache_dir}")
        except OSError as e:
            raise OSError(f"Не удалось создать директорию кэша {cache_dir}: {str(e)}") from e

        model_family = common_params["model_family"]

        # ----------------------------------------------
        # Регистрация модели (если указаны package/module/class)
        # ----------------------------------------------
        if all(k in common_params for k in ("package", "module", "model_class")):
            package = common_params["package"]
            module = common_params["module"]
            model_class = common_params["model_class"]

            if not all([package, module, model_class]):
                raise ValueError("package, module и model_class не могут быть пустыми")

            model_class_path = f"{package}.{module}:{model_class}"
            print(f"INFO: Регистрация модели '{model_family}': {model_class_path}")
            cls.register_model(model_family, model_class_path)

        # ----------------------------------------------
        # Формирование итоговых параметров конструктора
        # ----------------------------------------------
        # Передаем весь model_config в конструктор модели
        model_params: dict[str, Any] = {
            "model_config": model_config
        }

        print(f"INFO: Инициализируем модель: {common_params['model_name']}")

        try:
            return cls.get_model(model_family, model_params)
        except Exception as e:
            raise ValueError(f"Ошибка инициализации модели {model_family}: {str(e)}") from e
