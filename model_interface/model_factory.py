import importlib

class ModelFactory:
    """Фабрика для создания моделей."""
    _models = {}  # Словарь для хранения зарегистрированных моделей {model_name: model_class}

    @classmethod
    def register_model(cls, model_name, model_path):
        """Декоратор для регистрации модели по её пути."""
        module_path, class_name = model_path.split(":")
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        cls._models[model_name] = model_class

    @classmethod
    def get_model(cls, model_name, model_init_params=None):
        """Создает и возвращает экземпляр модели по имени."""
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' is not registered.")
        return cls._models[model_name](**model_init_params)
    