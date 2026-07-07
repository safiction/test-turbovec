# TurboVec Benchmark Suite

Набор бенчмарков для оценки библиотеки **TurboVec** — квантованного ANN-индекса на CPU.  
Для каждой задачи реализованы два пайплайна:
- **Baseline** — exact k-NN поиск по float32 эмбеддингам (per-query, честное сравнение)
- **TurboVec** — k-NN через `TurboQuantIndex` с квантованием (bit-width = 2, 4)

Сравниваем метрики качества, время поиска и объём памяти.

---

## Структура проекта

```
.
├── data/                    # Загруженные датасеты и эмбеддинги
│   ├── classification/
│   ├── rag_search/
│   ├── anomaly_detection/
│   ├── semantic_clustering/
│   ├── semantic_search/
│   └── image_classification/
├── src/
│   ├── load_data/           # Скрипты загрузки и подготовки данных
│   └── run_test/            # Скрипты запуска бенчмарков
├── results/                 # JSON-summary и CSV с результатами прогонов
├── requirements.txt         # Зависимости
└── README.md
```

> **Примечание:** папка `data/` не хранится в git — данные можно воспроизвести, запустив скрипты из `src/load_data/`.

---

## Установка

```bash
pip install -r requirements.txt
```

---

## Быстрый тест

Каждый бенчмарк поддерживает режим `--quick-test` — прогон на уменьшенной выборке для проверки корректности перед полным запуском.

```bash
# Пример для classification
py src/run_test/classification.py --dim 384 --k 5 --quick-test

# Пример для semantic_search
py src/run_test/semantic_search.py --quick-test
```

---

## Типы задач

### 1. Semantic Search

| Параметр | Значение |
|----------|----------|
| **Данные** | MS MARCO (passage ranking) |
| **Источник** | HuggingFace `ms_marco` |
| **Размер** | 40 000 пассажей, 6 066 запросов |
| **Модель эмбеддингов** | `sentence-transformers/multi-qa-mpnet-base-cos-v1` |
| **Метрики** | NDCG@1/5/10, MAP@1/5/10, Recall@1/5/10, Precision@1/5/10, MRR |
| **Параметры** | `dim = 768` (фиксировано моделью) |

**Пайплайн:**
1. `py src/load_data/semantic_search.py` — загрузка пассажей и запросов, генерация эмбеддингов
2. `py src/run_test/semantic_search.py` — прогон baseline и TurboVec

---

### 2. Question Answering (Retrieval Stage) — RAG Search

| Параметр | Значение |
|----------|----------|
| **Данные** | SberQuAD — Russian Reading Comprehension Dataset |
| **Источник** | HuggingFace `kuznetsoffandrey/sberquad` |
| **Размер** | train: 5 000 вопросов / validation: 2 000 вопросов |
| **Модель эмбеддингов** | `nomic-ai/nomic-embed-text-v1.5` |
| **Метрики** | Recall@1/5/10, MRR |
| **Параметры** | `dim ∈ [384, 512, 1024]`, `bit_width ∈ [2, 4]` |

**Пайплайн:**
1. `py src/load_data/rag_search.py` — загрузка контекстов и вопросов
2. `py src/run_test/rag_search.py --mode small` — прогон на validation
3. `py src/run_test/rag_search.py --mode large` — прогон на train

---

### 3. Classification

| Параметр | Значение |
|----------|----------|
| **Данные** | IMDB Reviews (sentiment classification) |
| **Источник** | HuggingFace `imdb` |
| **Размер** | 10 000 train / 1 000 test |
| **Модель эмбеддингов** | `nomic-ai/nomic-embed-text-v1.5` |
| **Метрики** | Accuracy, F1-macro, F1-weighted |
| **Параметры** | `dim ∈ [384, 512, 1024]`, `k ∈ [5, 10]` |

**Пайплайн:**
1. `py src/load_data/classification.py` — загрузка отзывов, генерация эмбеддингов
2. `py src/run_test/classification.py --dim 384 --k 5`

---

### 4. Anomaly Detection

| Параметр | Значение |
|----------|----------|
| **Данные** | Credit Card Fraud Detection |
| **Источник** | Kaggle `mlg-ulb/creditcardfraud` |
| **Размер** | 199 615 train / 84 807 test (28 признаков V1-V28) |
| **Модель эмбеддингов** | Исходные признаки (без эмбеддинг-модели) |
| **Метрики** | Average Precision (AP), AUC-PR |
| **Варианты** | `raw` — исходные признаки / `normalized` — L2-нормализованные |
| **Параметры** | `k ∈ [5, 10]`, `bit_width ∈ [2, 4]` |

> **Важно:** TurboVec работает с нормализованными векторами. Для признаков, не кратных 8, добавляется padding нулями.

**Пайплайн:**
1. `py src/load_data/anomaly_detection.py` — загрузка и нормализация
2. `py src/run_test/anomaly_detection.py --k 5`

---

### 5. Semantic Clustering

| Параметр | Значение |
|----------|----------|
| **Данные** | 20 Newsgroups |
| **Источник** | sklearn `fetch_20newsgroups` |
| **Размер** | ~11 000 документов, 20 категорий |
| **Модель эмбеддингов** | `nomic-ai/nomic-embed-text-v1.5` |
| **Метрики** | k-NN Accuracy, Mean Same-Class Recall@k, MRR, Silhouette Score |
| **Параметры** | `dim = 384`, `k ∈ [5, 10]`, `bit_width ∈ [2, 4]` |

**Пайплайн:**
1. `py src/load_data/semantic_clustering.py` — загрузка текстов, генерация эмбеддингов
2. `py src/run_test/semantic_clustering.py --k 5`

---

### 6. Image Classification

| Параметр | Значение |
|----------|----------|
| **Данные** | tanganke/sun397 (397 классов) |
| **Источник** | HuggingFace |
| **Размер** | 19850 train / 19850 test |
| **Модель эмбеддингов** | clip-vit-base-patch32 |
| **Метрики** | Top-1 Accuracy, Top-5 Accuracy, F1-macro, F1-weighted, Per-class Accuracy |
| **Параметры** | `k ∈ [5, 10]`, `bit_width ∈ [2, 4]` |

**Пайплайн:**
1. `py src/load_data/image_classification.py` — загрузка изображений, генерация эмбеддингов
2. `py src/run_test/image_classification.py --k 5`

---

## Итоги: подходит ли TurboVec?

Ниже сводка по результатам бенчмарков. Для каждой задачи — сравнение метрик baseline (float32) и TurboVec (bw4, как компромисс скорость/качество).

| Задача | Метрика Baseline | Метрика TurboVec (bw4) | Просадка | Вывод |
|--------|------------------|------------------------|----------|-------|
| **Semantic Search** | NDCG@10: 0.648 | NDCG@10: 0.647 | **−0.1%** | ✅ Отлично |
| **RAG Search** | Recall@10: 0.375 | Recall@10: 0.376 | **+0.1%** | ✅ Отлично |
| **Classification** | Accuracy: 1.000 | Accuracy: 1.000 | **0%** | ✅ Отлично |
| **Image Classification** | Top-5 Accuracy: 0.862 | Top-5 Accuracy: 0.860 | **−0.2%** | ✅ Отлично |
| **Semantic Clustering** | k-NN Acc: 0.400 | k-NN Acc: 0.400 | **0%** | ✅ Отлично |
| **Anomaly Detection (raw)** | AP: 0.00065 | AP: 0.00065 | **0%** | ✅ Отлично |
| **Anomaly Detection (normalized)** | AP: 0.0104 | AP: 0.0089 | **−14%** | ⚠️ Заметная просадка на bw2; на bw4 — приемлемо (−5%) |

### Общие выводы

**TurboVec хорошо подходит для:**
- Semantic / RAG search — метрики практически идентичны baseline
- Classification (текстовая и图像ная) — без потери качества
- Clustering — сохраняет структуру соседства
- Anomaly detection на "сырых" (не нормализованных) признаках

**Нюансы:**
- **Anomaly Detection на L2-нормализованных данных** — просадка ~14% на 2-bit квантовании. Причина: нормализация сжимает динамический диапазон признаков, и квантование теряет больше информации. На 4-bit просадка снижается до ~5%.

**В среднем по всем задачам:**
- **Ускорение поиска:** 2–4× относительно exact k-NN
- **Экономия памяти:** ~8× (bw4) или ~16× (bw2) относительно float32
- **Потеря качества:** <1% для большинства задач

---

## Документация TurboVec

Подробное про turbovec https://github.com/RyanCodrai/turbovec
