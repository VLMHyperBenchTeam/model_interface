import importlib
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Type, Protocol, Union

# Константы
DEFAULT_CACHE_DIR = "model_cache"
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_QWEN_MODEL = "Qwen2.5-VL-7B-Instruct"
QWEN_FAMILY_NAME = "Qwen2.5-VL"


class ModelInterface(Protocol):
    """Протокол для определения интерфейса модели."""
    
    def predict_on_image(self, image_path: str, question: str) -> str:
        """Предсказание на основе изображения и вопроса."""
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

    _models: Dict[str, Type[ModelInterface]] = {}
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
        cls, model_name: str, model_init_params: Optional[Dict[str, Any]] = None
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
    def get_registered_models(cls) -> Dict[str, Type[ModelInterface]]:
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
    def initialize_model(cls, model_config: Dict[str, Any]) -> ModelInterface:
        """Инициализирует и возвращает модель согласно конфигурации.
        
        Функция автоматически регистрирует модель в ModelFactory и создает экземпляр.
        Поддерживает как полную конфигурацию (с указанием package, module, model_class),
        так и упрощенную (только model_family для предварительно зарегистрированных моделей).

        Args:
            model_config: Словарь конфигурации модели. Обязательные ключи:
                - model_family: семейство модели (например, "Qwen2.5-VL")
                - model_name: имя модели (например, "Qwen2.5-VL-7B-Instruct")
                - cache_dir: директория для кэша модели
                - device_map: карта устройств для модели
                
                Опциональные ключи для регистрации новых моделей:
                - package: пакет с классом модели
                - module: модуль с классом модели  
                - model_class: имя класса модели

        Returns:
            ModelInterface: Инициализированный объект модели.

        Raises:
            KeyError: При отсутствии обязательных ключей в конфигурации.
            OSError: При ошибках создания директории кэша.
            ValueError: При ошибке регистрации или создания модели.
        """
        # Валидация конфигурации
        if not isinstance(model_config, dict):
            raise ValueError("model_config должен быть словарем")
            
        # Проверка обязательных ключей
        required_keys = {
            "model_family", 
            "cache_dir", 
            "model_name",
            "device_map"
        }
        if missing := required_keys - set(model_config):
            raise KeyError(f"Отсутствуют обязательные ключи: {missing}")

        # Валидация значений
        for key in required_keys:
            if not model_config[key]:
                raise ValueError(f"Ключ '{key}' не может быть пустым")

        # Создание директории для кэша
        cache_dir = Path(model_config["cache_dir"])
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"INFO: Директория кэша создана/проверена: {cache_dir}")
        except OSError as e:
            print(f"ERROR: Ошибка создания директории кэша {cache_dir}: {str(e)}")
            raise OSError(f"Не удалось создать директорию кэша {cache_dir}: {str(e)}") from e

        model_family = model_config["model_family"]
        
        # Регистрируем модель в фабрике, если указаны детали реализации
        if all(key in model_config for key in ["package", "module", "model_class"]):
            package = model_config["package"]
            module = model_config["module"]
            model_class = model_config["model_class"]
            
            # Валидация параметров регистрации
            if not all([package, module, model_class]):
                raise ValueError("Параметры регистрации (package, module, model_class) не могут быть пустыми")
                
            model_class_path = f"{package}.{module}:{model_class}"
            
            print(f"INFO: Регистрация модели {model_family}: {model_class_path}")
            cls.register_model(model_family, model_class_path)

        # Подготовка параметров модели
        model_params = {
            "model_name": model_config["model_name"],
            "system_prompt": model_config.get("system_prompt", ""),
            "cache_dir": str(cache_dir),
            "device_map": model_config["device_map"],
        }

        print(f"INFO: Инициализация модели: {model_config['model_name']}")
        
        # Создание экземпляра модели
        try:
            return cls.get_model(model_family, model_params)
        except Exception as e:
            print(f"ERROR: Ошибка создания модели {model_family}: {str(e)}")
            raise ValueError(f"Ошибка инициализации модели {model_family}") from e

    @classmethod
    def create_qwen_model_config(
        cls,
        model_name: str = DEFAULT_QWEN_MODEL,
        cache_dir: str = DEFAULT_CACHE_DIR, 
        device_map: str = DEFAULT_DEVICE_MAP,
        system_prompt: str = ""
    ) -> Dict[str, Any]:
        """Создает конфигурацию для модели Qwen2.5-VL с предустановленными параметрами.
        
        Модель Qwen2.5-VL автоматически регистрируется при импорте модуля model_qwen2_5_vl.models,
        поэтому здесь указываются только параметры инициализации.
        
        Args:
            model_name: Имя модели Qwen (по умолчанию "Qwen2.5-VL-7B-Instruct")
            cache_dir: Директория для кэша (по умолчанию "model_cache")
            device_map: Карта устройств (по умолчанию "auto")
            system_prompt: Системный промпт (по умолчанию пустой)
            
        Returns:
            Dict[str, Any]: Словарь конфигурации модели, готовый для передачи в initialize_model
            
        Raises:
            ValueError: При неверных входных параметрах.
        """
        # Валидация входных параметров
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name должен быть непустой строкой")
        if not cache_dir or not isinstance(cache_dir, str):
            raise ValueError("cache_dir должен быть непустой строкой")
        if not device_map or not isinstance(device_map, str):
            raise ValueError("device_map должен быть непустой строкой")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt должен быть строкой")
        
        # Импортируем модуль, чтобы запустить автоматическую регистрацию
        try:
            import model_qwen2_5_vl.models
            print("INFO: Модуль model_qwen2_5_vl.models успешно импортирован")
        except ImportError as e:
            print(f"WARNING: Не удалось импортировать model_qwen2_5_vl.models: {str(e)}. Модель может быть не зарегистрирована.")
        
        return {
            "model_family": QWEN_FAMILY_NAME,
            "model_name": model_name,
            "cache_dir": cache_dir,
            "device_map": device_map,
            "system_prompt": system_prompt,
        }

    @classmethod
    def initialize_qwen_model(
        cls,
        model_name: str = DEFAULT_QWEN_MODEL,
        cache_dir: str = DEFAULT_CACHE_DIR,
        device_map: str = DEFAULT_DEVICE_MAP,
        system_prompt: str = ""
    ) -> ModelInterface:
        """Упрощенная функция для инициализации модели Qwen2.5-VL.
        
        Объединяет создание конфигурации и инициализацию модели в один вызов.
        
        Args:
            model_name: Имя модели Qwen (по умолчанию "Qwen2.5-VL-7B-Instruct")
            cache_dir: Директория для кэша (по умолчанию "model_cache")
            device_map: Карта устройств (по умолчанию "auto")
            system_prompt: Системный промпт (по умолчанию пустой)
            
        Returns:
            ModelInterface: Инициализированный объект модели Qwen2.5-VL
            
        Raises:
            ValueError: При неверных входных параметрах или ошибке инициализации.
            
        Example:
            >>> model = ModelFactory.initialize_qwen_model("Qwen2.5-VL-7B-Instruct", device_map="cuda:0")
            >>> result = model.predict_on_image("image.jpg", "Что на картинке?")
        """
        try:
            config = cls.create_qwen_model_config(
                model_name=model_name,
                cache_dir=cache_dir,
                device_map=device_map,
                system_prompt=system_prompt
            )
            return cls.initialize_model(config)
        except Exception as e:
            print(f"ERROR: Ошибка инициализации модели Qwen: {str(e)}")
            raise ValueError(f"Не удалось инициализировать модель Qwen: {str(e)}") from e


def load_prompt(prompt_path: Union[str, Path]) -> str:
    """Загружает промпт из файла с проверкой существования.

    Args:
        prompt_path: Путь к файлу с промптом (строка или Path объект).

    Returns:
        str: Текст промпта.

    Raises:
        ValueError: При неверном типе пути.
        FileNotFoundError: Если файл не существует.
        OSError: При ошибке чтения файла.
    """
    # Валидация входного параметра
    if not prompt_path:
        raise ValueError("prompt_path не может быть пустым")
        
    # Преобразование в Path объект
    if isinstance(prompt_path, str):
        prompt_path = Path(prompt_path)
    elif not isinstance(prompt_path, Path):
        raise ValueError("prompt_path должен быть строкой или Path объектом")
    
    # Проверка существования файла
    if not prompt_path.exists():
        raise FileNotFoundError(f"Файл промпта не найден: {prompt_path}")
    
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Путь не является файлом: {prompt_path}")
    
    # Загрузка файла
    try:
        content = prompt_path.read_text(encoding="utf-8")
        print(f"INFO: Промпт успешно загружен из файла: {prompt_path}")
        return content
    except OSError as e:
        print(f"ERROR: Ошибка чтения файла промпта {prompt_path}: {str(e)}")
        raise OSError(f"Не удалось прочитать файл промпта {prompt_path}: {str(e)}") from e
