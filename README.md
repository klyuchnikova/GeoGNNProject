# GeoGNNProject — Graph-Flashback

Ветка содержит воспроизводимый pipeline рекомендации следующего POI на основе **Graph-Flashback**:

```text
Gowalla / Foursquare
        ↓
хронологическая подготовка данных
        ↓
spatial-temporal knowledge graph (STKG)
        ↓
TransE
        ↓
POI-transition graph + user–POI preference graph
        ↓
RNN + graph propagation + flashback aggregation
        ↓
Acc@1/5/10, MAP@5/10, MRR
```

Проект сохраняет исходные исследовательские материалы:

- `GeoGNNProject.pdf`;
- `gowalla_validated_metadata_eda.ipynb`;
- `foursquaregraphs-eda.ipynb`.

## Что исправлено относительно первого запуска

Первый Kaggle-run использовал более тяжёлую модифицированную конфигурацию и выбирал checkpoint по validation loss. В этой версии основной конфиг приближен к оригинальному Graph-Flashback:

- `RNN`, hidden size `10`;
- `learning_rate=0.01`, batch size `200`;
- scheduler на эпохах `20/40/60/80`;
- checkpoint выбирается по validation `MRR`;
- temporal/spatial flashback использует coordinate-space distance и `lambda_s=1000`;
- дополнительные spatial/friend GCN-ветви выключены в основном запуске;
- graph propagation без дополнительной trainable projection;
- spatial relation STKG строится по **k ближайшим POI**, симметрично;
- POI не фильтруются по будущим validation/test посещениям (`min_poi_visits=1`);
- STKG строится только по train split;
- отдельный `70/10/20` режим используется для выбора модели, а `80/20` — для финального paper-compatible запуска.

## Метрики

Для каждого следующего check-in имеется один правильный POI:

- `Acc@1`, `Acc@5`, `Acc@10` — доля случаев, когда правильный POI попал в top-k;
- `MAP@5`, `MAP@10` — `1/rank`, если правильный POI вошёл в top-k, иначе `0`;
- `MRR` — среднее `1/rank` без отсечения;
- дополнительно сохраняются macro-user версии метрик.

Основные метрики оригинальной статьи: `Acc@1/5/10` и `MRR`. Значения порядка `0.1–0.4` нормальны для full-vocabulary next-POI ranking; `0.8 Acc@1` для выбора одного POI из тысяч обычно означает другую постановку или утечку данных.

## Структура

```text
configs/                         YAML-конфигурации
flashback/data/                  adapters, выбор города, preprocessing, sequences
flashback/kge/                   STKG, TransE, построение KGE-графов
flashback/model/                 Graph-Flashback
flashback/evaluation/            ranking-метрики
flashback/analysis/              автоматические визуализации
notebooks/kaggle_graph_flashback.ipynb
scripts/make_synthetic_data.py   локальный smoke-test
scripts/apply_clean_update.py    удаление файлов старого громоздкого overlay
scripts/verify_merge.py          проверка обязательных файлов
tests/                           unit и end-to-end smoke tests
```

Папки `data/`, `checkpoints/` и `artifacts/` создаются только во время запуска и не хранятся в Git.

## Обновление локальной ветки на Windows

Распакуйте архив поверх корня `flashback-branch`, затем выполните:

```powershell
cd "D:\projects\hse sla\GeoGNNProject"
git checkout flashback-branch

python .\scripts\apply_clean_update.py
python -m pip install -r requirements.txt
python .\scripts\verify_merge.py

New-Item -ItemType Directory -Force ".pytest_tmp"
python -m pytest -q --basetemp="$PWD\.pytest_tmp"
Remove-Item -Recurse -Force ".pytest_tmp"
```

После успешной проверки:

```powershell
git add -A
git commit -m "Clean and fix Graph-Flashback reproduction"
git push origin flashback-branch
```

`apply_clean_update.py` не удаляет PDF, два EDA-ноутбука и готовые experiment artifacts с реальными результатами.

## Kaggle: рекомендуемый порядок

В Notebook settings включите **Internet** и **GPU T4**. Первый блок всегда клонирует нужную ветку:

```python
%cd /kaggle/working
!rm -rf GeoGNNProject
!git clone --depth 1 --branch flashback-branch --single-branch \
    https://github.com/klyuchnikova/GeoGNNProject.git
%cd /kaggle/working/GeoGNNProject
!git branch --show-current
!git log -1 --oneline
```

Не переустанавливайте предустановленный CUDA PyTorch:

```python
!grep -v '^torch' requirements.txt > /tmp/requirements-kaggle.txt
!python -m pip install -q -r /tmp/requirements-kaggle.txt
```

### 1. Smoke-test

```python
!python scripts/make_synthetic_data.py
!python -m flashback.pipeline --config configs/gowalla_smoke.yaml --stage all
!python -m pytest -q --basetemp=/kaggle/working/pytest_tmp
```

### 2. Development run с validation

Запускайте по стадиям, чтобы не потерять результат при ограничении Kaggle-сессии:

```python
CONFIG = "configs/gowalla_auto.yaml"
```

```python
!python -m flashback.pipeline --config $CONFIG --stage download
!python -m flashback.pipeline --config $CONFIG --stage prepare
!python -m flashback.pipeline --config $CONFIG --stage stkg
!python -m flashback.pipeline --config $CONFIG --stage kge
!python -m flashback.pipeline --config $CONFIG --stage graphs
!python -m flashback.pipeline --config $CONFIG --stage train
!python -m flashback.pipeline --config $CONFIG --stage analyze
```

`gowalla_auto.yaml` ранжирует крупные metro areas по совокупности eligible users, eligible check-ins, POI diversity и временному покрытию. Выбранный город сохраняется в `data/processed/city_ranking.csv` и manifest.

### 3. Финальный 80/20 запуск

После выбора города и параметров укажите этот город явно в `configs/gowalla_paper.yaml`, удалите старые generated artifacts либо смените output paths, затем повторите pipeline с:

```python
CONFIG = "configs/gowalla_paper.yaml"
```

В paper-режиме test не используется для выбора checkpoint: модель обучается все заданные эпохи, затем test оценивается один раз.

## Результаты

После запуска создаются:

```text
artifacts/results/*_metrics.json
artifacts/results/*_history.csv
artifacts/results/metrics_comparison.csv
artifacts/predictions/*_test.parquet
artifacts/figures/*.png
data/processed/*_manifest.json
data/processed/city_ranking.csv
data/kge/stkg_manifest.json
data/kge/transe_history.json
data/graphs/graph_manifest.json
checkpoints/*_best.pt
```

Обязательно сравнивайте Graph-Flashback с:

- global popularity;
- personal popularity.

Если модель проигрывает personal popularity, запуск нельзя считать успешным — нужно проверить learning curves, выбранный checkpoint, STKG/KGE и параметры конфигурации.

## Foursquare

Модель и метрики не меняются. Требуется только положить TSMC2014 CSV в:

```text
data/raw/foursquare/dataset_TSMC2014_NYC.csv
data/raw/foursquare/dataset_TSMC2014_TKY.csv
```

и выбрать конфиг:

```bash
python -m flashback.pipeline --config configs/foursquare_nyc.yaml --stage all
# или
python -m flashback.pipeline --config configs/foursquare_tky.yaml --stage all
```

У TSMC2014 нет Gowalla friendship graph, поэтому friendship relation отключена конфигом.

## Воспроизводимость и ограничения

- Случайный seed фиксируется.
- Все ID пользователей и POI формируются единообразно для sequence data, STKG и графов.
- Validation/test не используются при построении STKG.
- Полный Gowalla run может не завершиться в одной Kaggle-сессии; поэтому pipeline разделён на стадии.
- Метрики одного города нельзя напрямую сравнивать с опубликованными результатами на полном обработанном Gowalla.
