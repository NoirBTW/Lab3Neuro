# Лабораторная работа 3 — классификация насекомых

**Смирнов Александр Сергеевич.** Fine-tuning двух семейств timm на CPU, подбор гиперпараметров, экспорт ONNX и локальное приложение Gradio.

На отложенном тесте: **ResNet18 — 25/25 (100%)**, **MobileNetV3 — 23/25 (92%)**. Выбор ResNet18 сделан по validation до просмотра test. Маленький тест не доказывает безошибочную работу на новых фотографиях.

[Отчёт Word](Лабораторная%203%20—%20Смирнов%20Александр%20Сергеевич.docx) · [Исполненный notebook](experiments/notebooks/lab3.ipynb) · [Сравнение с образцом](Сравнение%20с%20образцом.md)

## Структура

Две рабочие папки: `experiments` (эксперименты и дообучение) и `app` (веб-приложение).
Внутри experiments используется структура по принципам Data Science Cookiecutter:

```text
experiments/
  data/raw/                 # отобранные исходные веб-фотографии
  data/processed/split.csv   # фиксированное разделение с SHA256 и группами авторов
  data/sources.csv           # авторы, лицензии, страницы и URL каждого фото
  configs/                  # конфигурации четырёх запусков
  notebooks/lab3.ipynb       # исполненный notebook с логами и графиками
  models/                   # checkpoint-файлы после повторного обучения
  reports/                  # метрики, кривые, матрицы ошибок и анализ
  train.py                  # TrainConfig dataclass, train/test/export
  best_config.json           # выбранная конфигурация
  requirements.txt
app/
  app.py                    # Gradio, только ONNX Runtime на CPU
  preprocessing.py          # единая предобработка с validation/test
  model.onnx                # готовые экспортированные веса
  model.json                # классы, размер, mean/std и конфигурация
  requirements.txt
```

## Данные

116 самостоятельно отобранных фотографий из Wikimedia Commons: 36 `butterfly` (Danaus plexippus), 39 `dragonfly` (Anax imperator), 41 `ladybird` (Coccinella septempunctata).
Это собственная подборка из открытых источников, **не собственная фотосъёмка** и не готовый чужой ML-датасет. Подробности и ограничения: [data/README](experiments/data/README.md); авторство всех изображений: [ATTRIBUTION](experiments/data/ATTRIBUTION.md).

Разделение по авторам исключает попадание снимков одного автора в разные части: train 69, validation 22, test 25.
При первом запуске интернет нужен для загрузки исходных pretrained весов timm. Для готового приложения интернет не требуется.

## Окружение экспериментов

Python 3.12, команды из корня репозитория. Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name neuro-lab --display-name "Lab3 Neuro"
```

Linux/macOS: используйте `python3.12` и `.venv/bin/python` вместо Windows-путей.
Прямые зависимости зафиксированы в requirements; полный снимок среды исполнения находится в `experiments/requirements-lock.txt`.

## Эксперименты и повторение

Откройте `experiments/notebooks/lab3.ipynb` в Jupyter или VS Code, выберите kernel `Lab3 Neuro` и выполните все ячейки.
Notebook содержит четыре настоящих обучения (две модели × два learning rate), выбор по validation, итоговый test, экспорт и повтор лучшего запуска.
Автоматическое исполнение того же notebook:

```powershell
.\.venv\Scripts\python.exe -m experiments.execute_notebook
```

Повторить только лучшую модель и экспортировать её:

```powershell
.\.venv\Scripts\python.exe -m experiments.train --config experiments/best_config.json
```

Имя checkpoint соответствует `run_name` в конфигурации. Команды оценки и экспорта:

```powershell
.\.venv\Scripts\python.exe -m experiments.train --test experiments/models/resnet18_lr001.pt
.\.venv\Scripts\python.exe -m experiments.train --export experiments/models/resnet18_lr001.pt
```

Checkpoint-файлы восстанавливаются обучением; готовый `app/model.onnx` включён в репозиторий.
`--prepare` пересоздаёт split с seed=42 из raw и sources.csv. Для точного повторения используйте включённый split.csv.
`collect_data.py` позволяет повторить поиск, но выдача Commons со временем меняется; фиксированная версия данных включена в Git.
Решения визуальной фильтрации находятся в `rejected.csv` и `curate_data.py`.

## Локальное приложение

Отдельное лёгкое окружение не требует PyTorch или timm:

```powershell
py -3.12 -m venv app/.venv
.\app\.venv\Scripts\python.exe -m pip install -r app/requirements.txt
.\app\.venv\Scripts\python.exe -m app.app
```

Откройте http://127.0.0.1:7860, загрузите фотографию и нажмите Submit.
Приложение выводит три вероятности и наиболее вероятный класс. Это softmax-оценки, а не калиброванные гарантии.
Оно всегда выбирает один из трёх известных видов и не умеет распознавать «неизвестный класс».
Фото остаётся в локальном приложении: `share=False`, адрес привязан к `127.0.0.1`.

Предобработка общая для оценки и приложения: EXIF transpose, RGB, центральный квадрат с BICUBIC 160×160, масштаб 1/255, mean/std ImageNet, CHW float32. Аугментации выполняются только при обучении.

## Методика и результаты

Python random, NumPy и PyTorch зафиксированы seed=42; детерминированные операции, num_workers=0, два CPU-потока.
TrainConfig — Python dataclass. Первые две эпохи обучается классификатор, затем вся сеть с в 10 раз меньшим шагом backbone. Замороженный BatchNorm не обновляет статистики.
AdamW, 6 эпох, batch 8; шаг головы 0.001 или 0.0003. Лучшая эпоха и конфигурация определяются по validation accuracy, при равенстве по validation loss. Test не используется для подбора.

Итоговые числа и их интерпретация: [анализ](experiments/reports/analysis.md), [выбор модели](experiments/reports/selection.json), [тест](experiments/reports/tests.json), [проверка ONNX](experiments/reports/export_verification.json), [повторяемость](experiments/reports/reproducibility.json).
Маленький тест и поиск по названию вида ограничивают внешнюю применимость; метрики не доказывают качество на произвольных снимках.

## Источники и лицензии

- [timm ResNet18](https://huggingface.co/timm/resnet18.a1_in1k)
- [timm MobileNetV3 Small](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- [timm quickstart](https://huggingface.co/docs/timm/quickstart)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [Wikimedia Commons reuse](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia)

Исходный код проекта — MIT (LICENSE). Изображения сохраняют индивидуальные лицензии в ATTRIBUTION.md; MIT на них не распространяется.
Исходные веса timm распространяются под Apache-2.0 согласно model cards; условия и уведомления приведены в `app/NOTICE.md`.
