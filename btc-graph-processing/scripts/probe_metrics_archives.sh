#!/usr/bin/env bash
# Проверка доступности daily-архивов деривативных метрик Binance USD-M.
#
# Нужна ровно для одного вопроса: с какой даты у монеты вообще есть файлы
# метрик. Ответ идёт в SymbolSpec.metrics_start — заводя новую монету,
# прогони это по её кандидатам в даты листинга, а не полагайся на дату
# листинга спота: у метрик своя граница и свои дыры.
#
# Выводы прошлого прогона зафиксированы в docs/tz_deriv_ingest_14-08-26.md,
# §0.3 — скрипт лежит здесь, чтобы их можно было перепроверить, а не
# переоткрывать. До 2026-08-15 он валялся в корне репозитория как test.sh.
#
#   bash scripts/probe_metrics_archives.sh
#
# 200 — файл есть, 404 — нет.
set -u

probe() {
  u="https://data.binance.vision/data/futures/um/daily/metrics/$1/$1-metrics-$2.zip"
  echo "$(curl -s -o /dev/null -w '%{http_code}' "$u")  $u"
}

for s in ${SYMBOLS:-BNBUSDT XRPUSDT ADAUSDT}; do
  for d in ${DAYS:-2021-06-01 2021-11-01 2021-12-01}; do
    probe "$s" "$d"
  done
done
