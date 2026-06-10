# Запуск Graph-Flashback в Kaggle

## 1. Создание notebook

1. Создайте новый Kaggle Notebook.
2. В `Settings` включите `Internet`.
3. Выберите GPU accelerator, если он доступен.
4. Загрузите этот репозиторий как ZIP или клонируйте ветку в `/kaggle/working`.

```bash
%cd /kaggle/working/GeoGNNProject
!python -m pip install -q -r requirements.txt
```

Kaggle уже содержит PyTorch; если pip пытается заменить совместимую CUDA-сборку, удалите строку `torch` из временной копии `requirements.txt` и используйте предустановленную версию.

## 2. Полный Gowalla run

Все исходные данные загружаются внутри Kaggle:

```bash
!python -m flashback.pipeline \
  --config configs/gowalla_auto.yaml \
  --stage download
```

Команда получает:

- официальный SNAP check-in файл;
- официальный SNAP friendship graph;
- `gowalla_spots_subset1.csv` через Figshare API, если файл доступен.

Далее можно выполнить этапы отдельно, чтобы сохранять checkpoint после каждого шага:

```bash
!python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage prepare
!python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage stkg
!python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage kge
!python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage graphs
!python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage train
!python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage analyze
```

Или одной командой:

```bash
!python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage all
```

## 3. Где лежат результаты

```text
data/processed/city_ranking.csv
data/kge/stkg_manifest.json
data/kge/transe_best.pt
data/graphs/*.npz
checkpoints/*_best.pt
artifacts/results/*_metrics.json
artifacts/results/*_history.csv
artifacts/predictions/*_test.parquet
artifacts/figures/*.png
```


## 4. Сохранение результата Kaggle-сессии

```bash
!zip -r /kaggle/working/graph_flashback_output.zip \
  artifacts checkpoints data/processed data/kge data/graphs configs
```

Скачайте `graph_flashback_output.zip` из панели Output либо выполните `Save Version`, чтобы артефакты сохранились как notebook output.

## 5. Восстановление после остановки сессии

Загрузите предыдущий `graph_flashback_output.zip` как Kaggle Dataset и распакуйте:

```bash
!unzip -q /kaggle/input/<your-output-dataset>/graph_flashback_output.zip \
  -d /kaggle/working/GeoGNNProject
```

После этого продолжайте с нужного этапа, например `--stage train`.

## 6. Foursquare

Добавьте Kaggle dataset `chetanism/foursquare-nyc-and-tokyo-checkin-dataset` в notebook или загрузите его CLI:

```bash
!python scripts/download_foursquare.py
```

Проверьте фактическое имя файла и при необходимости исправьте только `data.raw_checkins` в YAML. Затем:

```bash
!python -m flashback.pipeline --config configs/foursquare_nyc.yaml --stage all
# или
!python -m flashback.pipeline --config configs/foursquare_tky.yaml --stage all
```

Код модели не меняется. В TSMC friendship graph отсутствует, поэтому Foursquare preset устанавливает `use_friend_graph: false`.

## 7. Быстрая проверка перед полным запуском

```bash
!python scripts/make_synthetic_data.py
!python -m flashback.pipeline --config configs/gowalla_smoke.yaml --stage all
!pytest -q
```

## 8. OOM и производительность

- Уменьшите `kge.batch_size` и `train.batch_size`.
- Уменьшите `graphs.score_chunk_size`; это снижает пиковую память при exact top-k KGE scoring.
- Не уменьшайте `transition_topk` только для получения лучших цифр без validation-сравнения.
- Для smoke-test допустимо уменьшить embedding/hidden dimensions и epochs; для итогового эксперимента используйте `gowalla_auto.yaml`.
- Графы хранятся как sparse CSR `.npz`, поэтому их не следует преобразовывать в dense matrices.
