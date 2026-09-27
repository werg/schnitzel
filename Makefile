.PHONY: test lint smoke doctor spark-build

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

smoke:
	schnitz train --config configs/tiny_cpu.yaml --output runs/tiny-smoke --steps 5
	schnitz evaluate --run runs/tiny-smoke --count 2

doctor:
	schnitz doctor

spark-build:
	./scripts/spark.sh build

trajectory-smoke:
	schnitz launch --recipe recipes/offline_smoke.yaml --output runs/trajectory-smoke

train-starter:
	./scripts/start_spark.sh --recipe recipes/starter.yaml --output runs/starter
