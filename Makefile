.PHONY: install download prepare stkg kge graphs train analyze all test smoke
install:
	python -m pip install -r requirements.txt
download:
	python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage download
prepare:
	python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage prepare
stkg:
	python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage stkg
kge:
	python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage kge
graphs:
	python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage graphs
train:
	python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage train
analyze:
	python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage analyze
all:
	python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage all
test:
	python -m pytest -q
smoke:
	python scripts/make_synthetic_data.py
	python -m flashback.pipeline --config configs/gowalla_smoke.yaml --stage all
