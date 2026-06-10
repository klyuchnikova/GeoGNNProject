# Методология

## Данные и выбор города

Основной Gowalla источник — SNAP check-ins и friendship graph. Город не задаётся случайно: `rank_cities` одним проходом считает объём check-ins, пользователей, пользователей после `min_checkins`, POI и временной охват для фиксированных metro bounding boxes. Решение принимается до обучения и сохраняется в `city_ranking.csv`.

Spot metadata присоединяется только для EDA, категорий и контроля качества. Категория и название POI не подаются в модель и не изменяют архитектуру Graph-Flashback.

## Сплиты

Основной preset использует хронологический `70/10/20` train/validation/test для каждого пользователя. STKG строится исключительно по train. Validation выбирает checkpoint, test оценивается после выбора модели.

`gowalla_paper.yaml` предоставляет `80/20`, близкий к исходной статье, но он не является предпочтительным режимом выбора гиперпараметров.

Последовательности имеют target mask: вход может включать прошлый контекст из предыдущего split, однако loss и метрики считаются только для targets текущего split.

## STKG

Сущности:

- пользователи;
- POI.

Отношения:

- `visits`: user → POI;
- `temporal`: последовательный POI → POI;
- `spatial`: POI → POI в радиусе;
- `friend`: user → user.

Все triplets выводятся только из train-части. Пространственные соседи ищутся BallTree с haversine distance.

## TransE

Встроенная реализация использует relation-aware negative sampling: corrupted entity всегда берётся из допустимого домена конкретного отношения. Objective — margin ranking loss. Entity embeddings нормализуются после optimizer step. Лучший checkpoint выбирается по validation triplets, которые являются случайным подмножеством train-derived STKG и не содержат test check-ins.

## KGE-derived graphs

После TransE для каждой головы вычисляется exact top-k по `exp(-L1 distance)`:

- relation `temporal` создаёт POI transition graph;
- relation `visits` создаёт user–POI preference graph;
- relation `friend` создаёт user graph.

Расчёт выполняется chunks, не создавая полную матрицу scores на CPU. Графы сохраняются как sparse CSR.

## Graph-Flashback

Модель сохраняет компоненты оригинального подхода:

1. POI embeddings;
2. graph propagation по KGE transition graph;
3. дополнительный spatial graph;
4. recurrent layer RNN/GRU/LSTM;
5. временной flashback coefficient;
6. пространственный flashback coefficient;
7. user–POI preference similarity;
8. user embedding, включая опциональный friend graph;
9. full-vocabulary next-POI classification.

Flashback weights векторизованы, но формула не упрощена. Friend graph path исправлен: graph-aware user embedding не перезаписывается обычным embedding.

## Метрики

Для единственного релевантного next POI:

- `Acc@k = I(rank ≤ k)`;
- `AP@k = 1/rank`, если `rank ≤ k`, иначе 0;
- `MAP@k` — среднее `AP@k`;
- `MRR` — среднее `1/rank`.

Считаются `Acc@1/5/10`, `MAP@5/10`, `MRR`, micro average и macro-user average. Ties разрешаются детерминированно по возрастающему POI id, а не оптимистичным best-rank.
