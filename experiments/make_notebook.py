"""Build the executable experiment notebook. Run from the repository root."""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# Лабораторная работа 3 — Fine-tuning
**Смирнов Александр Сергеевич**

Классификация самостоятельно отобранных фотографий трёх видов насекомых из Wikimedia Commons.
Авторы и лицензии приведены в `experiments/data/sources.csv`. Фотографии не сделаны студентом.
Сравниваем ResNet18 и MobileNetV3 Small из timm на CPU. Все результаты ниже вычисляются кодом.
Ноутбук нужно запускать из клонированного репозитория; исходные изображения уже включены.
''')
code('''from pathlib import Path
import os, sys, json, csv, platform
from dataclasses import asdict, replace
import numpy as np
from IPython.display import display, Image as DisplayImage, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'experiments/train.py').exists())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from experiments.train import TrainConfig, prepare_split, train, test_checkpoint, export_checkpoint, set_seed
set_seed(42)
print('Python', platform.python_version(), 'CPU', platform.processor())
import torch, timm
print('PyTorch', torch.__version__, 'timm', timm.__version__, 'CUDA', torch.cuda.is_available())''')
md('''## Данные и протокол
Три класса: butterfly — Danaus plexippus, dragonfly — Anax imperator, ladybird — Coccinella septempunctata.
Это классификация между тремя конкретными видами, а не универсальный определитель всех насекомых.
Из поиска исключены рисунки, личинки, коллекционные экземпляры и неоднозначные сцены; затем выполнен визуальный контроль.
Разделение около 60/20/20 группируется по автору: один автор не встречается в разных частях.
Метаданные и состав частей фиксируются до обучения. Тест не участвует в подборе.
''')
code('''rows = list(csv.DictReader(open('experiments/data/processed/split.csv', encoding='utf-8')))
classes = sorted({r['label'] for r in rows})
for c in classes:
    print(c, {s: sum(r['label']==c and r['split']==s for r in rows) for s in ['train','val','test']})
groups = {s: {r['group'] for r in rows if r['split']==s} for s in ['train','val','test']}
assert not groups['train'] & groups['val'] and not groups['train'] & groups['test'] and not groups['val'] & groups['test']
assert len({r['sha256'] for r in rows}) == len(rows)
display(DisplayImage(filename='experiments/reports/figures/examples.png'))''')
md('''## Архитектуры и стратегия
ResNet18 использует остаточные связи; MobileNetV3 Small — компактные блоки с depthwise-свёртками.
Обе сети загружаются с `pretrained=True`, классификатор заменяется на три выхода.
Первые две эпохи обучается голова, backbone и статистики BatchNorm заморожены. Затем размораживается вся сеть.
AdamW: шаг backbone в 10 раз меньше шага головы, weight decay 1e-4.
При переходе создаётся новый оптимизатор для нового этапа.
Аугментации только train: случайный crop 75–100% площади, горизонтальное отражение, умеренный ColorJitter.
Eval и приложение вызывают одну функцию: EXIF→RGB→центральный квадрат BICUBIC 160×160→ImageNet normalization→CHW float32.
''')
md('''## Подбор гиперпараметров
Фиксированная сетка: две архитектуры × два шага головы (0.001 и 0.0003), backbone=head/10.
Каждый запуск: 6 эпох, batch=8, seed=42, 2 CPU threads. Такая сетка ограничивает стоимость CPU-эксперимента.
Для каждого запуска выбирается эпоха с максимальной validation accuracy, при равенстве — минимальным validation loss.
Затем тем же критерием выбирается запуск в семействе и общая лучшая модель. Test открывается после выбора.
''')
code('''configs = []
for family, model in [('resnet18','resnet18.a1_in1k'), ('mobilenetv3','mobilenetv3_small_100.lamb_in1k')]:
    for lr, tag in [(1e-3,'001'), (3e-4,'0003')]:
        configs.append(TrainConfig(model=model, lr_head=lr, lr_backbone=lr/10, run_name=f'{family}_lr{tag}'))
Path('experiments/configs').mkdir(exist_ok=True)
runs = []
for config in configs:
    Path(f'experiments/configs/{config.run_name}.json').write_text(json.dumps(asdict(config), indent=2))
    runs.append(train(config))
Path('experiments/reports/search.json').write_text(json.dumps(runs, indent=2))''')
code('''def rank(r): return (r['best_validation_accuracy'], -r['best_validation_loss'])
winners = [max([r for r in runs if r['config']['model']==m], key=rank) for m in sorted({r['config']['model'] for r in runs})]
best = max(winners, key=rank)
for r in runs:
    print(r['config']['run_name'], 'val accuracy', round(r['best_validation_accuracy'],4),
          'val loss', round(r['best_validation_loss'],4), 'seconds', round(r['seconds'],1), 'parameters', r['parameters'])
    display(DisplayImage(filename=f"experiments/reports/figures/{r['config']['run_name']}.png"))
print('SELECTED BEFORE TEST:', best['config']['run_name'])
Path('experiments/reports/selection.json').write_text(json.dumps({'winner':best['config']['run_name'], 'families':[r['config']['run_name'] for r in winners]},indent=2))''')
md('''## Итоговая оценка
Accuracy = число правильных ответов / число изображений. В матрице ошибок строки — истинный класс, столбцы — прогноз.
Для каждого семейства оцениваем только выбранный по validation checkpoint. Базовый уровень для сбалансированных трёх классов около 1/3.
При небольшом тесте одна ошибка существенно меняет процент; результат нельзя переносить на все фотографии насекомых.
''')
code('''tests={}
for r in winners:
    name=r['config']['run_name']
    tests[name]=test_checkpoint(ROOT/f'experiments/models/{name}.pt')
    print(name, json.dumps(tests[name], ensure_ascii=False))
    display(DisplayImage(filename=f'experiments/reports/figures/{name}_confusion.png'))
Path('experiments/reports/tests.json').write_text(json.dumps(tests,indent=2))''')
md('''## ONNX и воспроизводимость
Экспортируем выбранную до просмотра test модель, opset=17, динамический batch, CPUExecutionProvider.
Проверяем структуру ONNX и численную близость логитов PyTorch и ONNX Runtime.
Повтор лучшего запуска с тем же seed проверяет все кривые обучения и веса; время исполнения не сравнивается.
''')
code('''best_path=ROOT/f"experiments/models/{best['config']['run_name']}.pt"
metadata=export_checkpoint(best_path)
print(json.dumps(metadata, indent=2))
repeat=train(replace(TrainConfig(**best['config']), run_name=best['config']['run_name']+'_repeat'))
assert repeat['history'] == best['history']
a=torch.load(best_path,map_location='cpu',weights_only=False)['state_dict']
b=torch.load(ROOT/f"experiments/models/{repeat['config']['run_name']}.pt",map_location='cpu',weights_only=False)['state_dict']
assert all(torch.equal(a[k],b[k]) for k in a)
Path('experiments/reports/reproducibility.json').write_text(json.dumps({'seed':42,'histories_equal':True,'weights_equal':True,'run':best['config']['run_name']},indent=2))
print('Same seed: histories and all checkpoint tensors are exactly equal.')''')
md('''## Проверка приложения на тестовых фотографиях
Проверка ниже использует настоящие тестовые изображения, а не только случайный вход.
Классы и нормализация считываются из экспортированных метаданных.
''')
code('''from experiments.verify_export import verify
verification=verify(best_path)
print(json.dumps(verification, indent=2))''')
md('''## Выводы
Численные выводы, ограничения, разбор ошибок и сравнение с предоставленным образцом приведены в `experiments/reports/analysis.md` и отчёте Word.
Команда локального приложения из корня: `python -m app.app`; адрес http://127.0.0.1:7860.
Исходники экспериментов доступны через `python -m experiments.train --help`; лучшая конфигурация — `experiments/best_config.json`.
''')
notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'neuro-lab'},'language_info':{'name':'python','version':'3.12'}})
nb.write(notebook,ROOT/'experiments/notebooks/lab3.ipynb')
print('Notebook created')
