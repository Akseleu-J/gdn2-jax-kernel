# ⚡ GDN-2 Pallas — High-Performance Gated DeltaNet-2 for TPU v5e

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-green.svg)](https://github.com/google/jax)

**GDN-2 Pallas** — это компактная, высокопроизводительная реализация **Gated DeltaNet-2** (GDN-2) на базе **Pallas** (JAX), оптимизированная для **TPU v5e**. Пакет включает полностью слитый (fused) backward на Pallas, что даёт ускорение **до 12x в forward** и **до 8.5x в backward** по сравнению с эталонной реализацией на `jax.associative_scan`.

---

## 🚀 Основные возможности

- **🚄 Молниеносная скорость** — forward и backward полностью написаны на Pallas с использованием MXU-блоков TPU.
- **🧩 Простота использования** — одна функция `gdn2_pallas_forward_trainable` для обучения и `gdn2_pallas_forward` для инференса.
- **🔒 Численная стабильность** — встроенные защиты (clipping, `nan_to_num`) на всех этапах.
- **📦 Лёгкий вес** — всего три файла (fwd, bwd, pipeline) без лишних зависимостей.
- **🔬 Гибкость** — легко настраиваемые параметры (`BT`, `BC`, `CLIP`) через импорт констант.

---

## 📊 Бенчмарки (на TPU v5e‑8)

Я сравнили нашу реализацию с эталонной реализацией **GDN-2 на JAX `associative_scan`** (авторская реализация, широко используемая в исследовательских проектах). Замеры проводились на **batch_size = 8**, **seq_len = 8192**, **heads = 6**, **d_head = 128** (полный профиль обучения с backward).

| Режим | Эталонная реализация (ms) | GDN-2 Pallas (ms) | Ускорение |
|-------|---------------------------|-------------------|-----------|
| **Forward** (без backward) | 48 | **4** | **12.0x** |
| **Backward** (весь градиент) | 180 | **21** | **8.5x** |
| **Полный цикл** (fwd + bwd) | 228 | **25** | **9.1x** |

> 🔥 **Итог:** Наш код на Pallas превосходит эталонную реализацию по скорости более чем в **9 раз** в среднем, что позволяет ускорить обучение гибридных моделей (GDN-2 + Mamba-2) в **>6 раз** в реальном тренировочном цикле.(реализация Mamba-2 можете найти по этой ссылке [**Mamba-2 Pallas**](https://github.com/Akseleu-J/mamba2-pallas)

---

## 📦 Установка

### Установка из репозитория (рекомендуется)

```bash
pip install git+https://github.com/Akseleu-J/gdn2-pallas.git
```

---

## Установка в режиме разработки
```bash
git clone https://github.com/Akseleu-J/gdn2-pallas.git
cd gdn2-pallas
pip install -e .
```

---

## 🧠 Быстрый старт
```python
import jax
import jax.numpy as jnp
from gdn2_package import gdn2_pallas_forward_trainable

# Создаём случайные тензоры: (batch, seq_len, heads, d_head)
shape = (4, 2048, 6, 128)
q = jnp.ones(shape, dtype=jnp.float32)
k = jnp.ones(shape, dtype=jnp.float32)
v = jnp.ones(shape, dtype=jnp.float32)
w = jnp.ones(shape, dtype=jnp.float32)
b = jnp.ones(shape, dtype=jnp.float32)
g = jnp.ones(shape, dtype=jnp.float32)   # log‑decay
scale = 0.1

# Forward + Backward (через custom_vjp)
out, h_final = gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale)

# Только Forward (для инференса)
from gdn2_package import gdn2_pallas_forward
out_inf, _ = gdn2_pallas_forward(q, k, v, w, b, g, scale)

print(out.shape, h_final.shape)   # (4, 2048, 6, 128) (4, 6, 128, 128)
```
> **Примечание**: Функция автоматически подбирает устройство (CPU/GPU/TPU) через JAX. Для TPU v5e-8 убедитесь, что установлены правильные драйверы и jaxlib собран для TPU.

---

## 🔧 Настройка параметров ядра
Вы можете изменить глобальные константы, чтобы адаптировать производительность под свой бюджет памяти или размер головы.

```python
from gdn2_package import gdn2_utils   # если вы решите вынести константы в отдельный модуль
# (пока константы определены внутри gdn2_fwd, их можно переопределить)
import gdn2_package.gdn2_fwd as fwd
fwd.BT = 128   # уменьшить размер чанка, если не хватает памяти
fwd.BC = 64
fwd.CLIP = 1e3
```

---

## 📖 Архитектура пакета
Пакет состоит из трёх ключевых модулей:

**gdn2_fwd.py** — реализует все forward‑кернелы (A, B, C, D) и сборку полного forward‑пайплайна.

**gdn2_bwd.py** — содержит все backward‑кернелы (B1–B5), которые используются в обратном распространении.

**gdn2_pipeline.py** — объединяет forward и backward в единый custom_vjp, предоставляя функцию gdn2_pallas_forward_trainable.

Эта архитектура обеспечивает максимальную модульность и упрощает внесение изменений.

---

## 🧑‍💻 Автор
**Akseleu Omirbay** 
Проект создан для высокопроизводительных исследований на TPU v5e.
Вопросы и предложения приветствуются через Issues.

---

## 📄 Лицензия
Распространяется под лицензией **MIT**.

---
## ⭐ Поддержка
Если этот пакет помог вам в работе или исследовании, поставьте ⭐ на GitHub — это поможет другим исследователям найти его!
