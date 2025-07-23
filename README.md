# 🤖 Model Interface — Универсальный интерфейс для VLM-моделей

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv 0.8.2](https://img.shields.io/badge/dependencies-uv%200.8.2-blue)](https://github.com/astral-sh/uv)

**Model Interface** — это гибкий python-пакет для унификации работы с VLM-моделями для `VLMHyperBench`([ссылка](https://github.com/VLMHyperBenchTeam/VLMHyperBench)).

Может использоваться и в других проектах.

**Создана для:**
- Быстрой интеграция VLM-моделей (`база`: изображение, `prompt`: текст, `output`: текст)
- Единый интерфейс VLM для работы с разными фреймворками (`Hugging Face`, `vLLM`, `SGLang`)
- Осуществляет кэширование весов моделей

**В разработке:**
- Измерение времени инференса VLM и потребляемых ресурсов

## 🚀 Особенности

✨ **Основные возможности**:
- **Фабрика моделей** — динамическая регистрация VLM моделей на производство в `ModelFactory` из ваших python-пакетов всего за 2 строки кода
- **Универсальный интерфейс** — единые методы для работы со всеми VLM-моделями (унаследованными от `ModelInterface`)
- **Кэширование весов модели** — сохраняет веса модели локально в указанную вами папку `cache_dir` для повторного использования.
- **Совместимость** — работает с любыми фреймворками инференса через абстрактные классы `ModelFactory` и `ModelInterface`.

## 🏛️ Архитектура

- **ModelFactory**: Фабричный класс для регистрации и создания экземпляров моделей из python-пакетов.
- **ModelInterface**: Интерфейсный класс, который определяет стандартные методы для работы с VLM.

![Architecture](docs/diagrams/model_interface.svg)

## 📚 Документация

TODO: скоро здесь появится ссылка.

## 📦 Установка

### Установка из PyPI (рекомендуемый способ)
```bash
uv pip install model_interface
# или
pip install model_interface
```

### Установка из репозитория (Git)
*   **Для пользователей с `uv` и `pyproject.toml` (рекомендуемый способ)**:
    ```bash
    uv add git+https://github.com/VLMHyperBenchTeam/model_interface.git@main
    uv sync
    ```
*   **Для пользователей с `pip` или `uv` (прямая установка)**:
    ```bash
    pip install git+https://github.com/VLMHyperBenchTeam/model_interface.git@main
    # или
    uv pip install git+https://github.com/VLMHyperBenchTeam/model_interface.git@main
    ```

## 🛠 Использование

### 1. Регистрация модели

```python
from model_interface.model_factory import ModelFactory

# Регистрация новой модели
ModelFactory.register_model(
    model_name="my_custom_model",
    model_path="my_ml_models.module:MyModelClass"
)
```

### Quick Start — вложенная конфигурация

Для моделей, уже зарегистрированных при импорте собственного python-пакета, достаточно
передать **вложенный** словарь конфигурации в `ModelFactory.initialize_model`:

```python
from model_interface.model_factory import ModelFactory

config = {
    "common_params": {
        "model_family": "Qwen2.5-VL",
        "model_name": "Qwen2.5-VL-7B-Instruct",
        "cache_dir": "model_cache",
        "device_map": "auto",
        "system_prompt": "Ты — эксперт по документам",
    },
    # Параметры, специфичные для семейства/конкретной модели
    "specific_params": {
        "max_size": 2048 * 28 * 28,
        "min_size": 512 * 28 * 28,
    },
}

model = ModelFactory.initialize_model(config)
result = model.predict_on_image("invoice.jpg", "Опиши документ")
```

Если пакет модели предоставляет функцию-обёртку, то можно использовать её напрямую
(см. пример для Qwen ниже).

### 2. Создание экземпляра
```python
model = ModelFactory.get_model(
    model_name="my_custom_model",
    model_init_params={
        "system_prompt": "Ты — эксперт по документам",
        "cache_dir": "/tmp/model_cache"
    }
)
```

### 3. Получение предсказаний
```python
# Для одного изображения
result = model.predict_on_image(image_object, "Что изображено на фото?")

# Для нескольких изображений
results = model.predict_on_images([img1, img2], "Сравни эти документы")
```

## 📂 Структура проекта
```
src/
└── model_interface
    ├── __init__.py
    ├── model_factory.py  — Фабрика для регистрации и создания моделей
    ├── model_interface.py — Базовый абстрактный класс для всех моделей
    └── model_utils.py    — Утилиты для измерения производительности
tests/                — Unit-тесты
pyproject.toml        — Основная конфигурация проекта (сборка с помощью uv_build)
```

## 📦 Примеры реализации Python-пакетов для VLM

Этот пакет используется в следующих проектах:

### Простой пример реализации интерфейса с "шуточной" моделью

   Рекомендуется для первого ознакомления.

1. **[model_qwen2_vl_example](https://github.com/VLMHyperBenchTeam/model_qwen2_vl_example)**  
   
   Демонстрирует:
   - Регистрацию модели через `ModelFactory`
   - Реализацию абстрактного класса `ModelInterface`
   - Тестовый сценарий в `example.py`

### Примеры уже реализованных python-пакетов для VLM

1. **[model_qwen2-vl](https://github.com/VLMHyperBenchTeam/model_qwen2-vl)**

   Реальная реализация для семейства моделей Qwen2-VL:
   - Поддержка 2B/7B версий
   - Фреймворк Hugging Face
   - Обработка изображений с регулируемым разрешением
   - Примеры работы с документами (счета, накладные)

2. **[model_qwen2_5_vl](https://github.com/VLMHyperBenchTeam/model_qwen2.5-vl)**  
   Реальная реализация для семейства моделей Qwen2.5-VL:
   - Поддержка 3B/7B версий
   - Фреймворк Hugging Face
   - Обработка изображений с регулируемым разрешением
   - Примеры работы с документами (счета, накладные)
