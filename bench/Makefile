# Стенд сравнения Go и Python на сборке баров из тиков Bybit.
# Подробности и результаты — README.md.

GO      ?= /usr/local/go/bin/go
PY      ?= python3
TICKS   ?= ../btc-graph-processing/data/bybit/HYPEUSDT/ticks
MONTH   ?= $(TICKS)/2026-06.csv.gz
OUT     ?= /tmp/bench-ticks2bars
BIN     := ticks2bars

.PHONY: build bench bench-full floor clean

build:
	cd go && $(GO) build -o ../$(BIN) .

$(BIN): build

bench: $(BIN)
	@mkdir -p $(OUT)
	@echo "=== python (боевой код btcproc)"
	@$(PY) py/ticks2bars.py $(MONTH) --out $(OUT)/py.csv
	@echo "=== go"
	@./$(BIN) --out $(OUT)/go.csv $(MONTH)
	@echo "=== сверка"
	@$(PY) compare.py $(OUT)/py.csv $(OUT)/go.csv

# Вся история монеты: только месячные архивы. Дневные сюда не берутся —
# они перекрываются с месячными, и порядок по имени файла у них не
# хронологический.
bench-full: $(BIN)
	@mkdir -p $(OUT)
	@echo "=== python"
	@$(PY) py/ticks2bars.py $(sort $(wildcard $(TICKS)/????-??.csv.gz)) --out $(OUT)/py-full.csv
	@echo "=== go, 1 поток"
	@./$(BIN) --workers 1 --out $(OUT)/go-full.csv $(sort $(wildcard $(TICKS)/????-??.csv.gz))
	@echo "=== go, 6 потоков"
	@./$(BIN) --workers 6 --out $(OUT)/go-full6.csv $(sort $(wildcard $(TICKS)/????-??.csv.gz))
	@echo "=== сверка"
	@$(PY) compare.py $(OUT)/py-full.csv $(OUT)/go-full.csv
	@cmp $(OUT)/go-full.csv $(OUT)/go-full6.csv && echo "1 и 6 потоков: выход побайтово одинаков"

# Пол для любой реализации: сколько стоит одна распаковка.
floor:
	@time (cat $(sort $(wildcard $(TICKS)/????-??.csv.gz)) | gzip -dc | wc -c)

clean:
	rm -f $(BIN)
	rm -rf $(OUT)
