PY ?= python3
export PYTHONPATH := src

.PHONY: setup foundation validate forecast detect cases figures report all test clean

setup:                 ## зависимости
	$(PY) -m pip install -r requirements.txt

foundation:            ## необязательно: Chronos-Bolt (нужен доступ к huggingface.co)
	$(PY) -m pip install -r requirements-foundation.txt
	$(PY) -c "from huggingface_hub import snapshot_download as s; s('amazon/chronos-bolt-small', local_dir='models/chronos-bolt-small')"

validate:              ## проверка данных и справочника
	$(PY) -m sbd.cli validate

forecast:              ## бэктест прогнозов (Prophet ~1–1.5 ч на 2 ядрах, результаты кэшируются)
	$(PY) -m sbd.cli forecast

detect:                ## детекторы и полусинтетика (~15 мин)
	$(PY) -m sbd.cli detect

cases:                 ## паводки 2024 и новостной слой
	$(PY) -m sbd.cli cases

figures:
	$(PY) -m sbd.cli figures

report:                ## PDF отчёта и презентации из outputs/
	$(PY) report/build.py

all: validate forecast detect cases figures report

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf outputs/*
