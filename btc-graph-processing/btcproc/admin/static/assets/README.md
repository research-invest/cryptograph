# Статика админки

Всё, что грузит браузер, лежит здесь — админка не ходит в CDN ни за чем.
Она работает во внутреннем контуре, и внешний запрос там либо висит до
таймаута, либо не проходит вовсе: страница графа осталась бы пустой без
единой ошибки в логе сервера.

```
assets/css/app.css   свой стиль
assets/js/           сторонние библиотеки, ниже
```

Шрифты системные (`-apple-system`, `Segoe UI`, `Roboto`), веб-шрифтов нет —
это тоже был бы внешний запрос.

## Версии библиотек

| Файл | Версия | Где используется | Источник |
|---|---|---|---|
| `js/htmx.min.js` | 1.9.12 | `base.html`, все страницы | `https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js` |
| `js/cytoscape.min.js` | 3.30.2 | `graph.html` — граф состояний | `https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js` |
| `js/lightweight-charts.standalone.production.js` | 4.1.3 | `chart.html` — свечи и разметка | `https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js` |

Версию каждого файла видно в шапке самого файла — в имени она не дублируется
намеренно, иначе обновление тянуло бы за собой правку ссылки в шаблоне.

## Как обновлять

```bash
curl -fL -o btcproc/admin/static/assets/js/cytoscape.min.js \
  https://unpkg.com/cytoscape@<версия>/dist/cytoscape.min.js
```

После обновления — обязательно открыть обе страницы руками. Мажорные версии
обеих библиотек ломают API молча: у lightweight-charts 5.x вместо
`addCandlestickSeries()` общий `addSeries(CandlestickSeries, …)`, и график
просто не отрисовывается. Таблицу версий выше при этом править тоже нужно —
она единственное место, где записано, что именно здесь лежит.

Нужна только `standalone`-сборка lightweight-charts: обычная — ES-модуль и
глобального `LightweightCharts` не заводит.
