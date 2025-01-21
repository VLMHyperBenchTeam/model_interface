from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Абстрактный класс для всех моделей."""
    def __init__(self, model_name, system_prompt, cache_dir):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir

    @abstractmethod
    def predict_on_image(self, image, question):
        """Метод для предсказания по одному изображению."""
        pass

    @abstractmethod
    def predict_on_images(self, images, question):
        """Метод для предсказания по нескольким изображениям."""
        pass