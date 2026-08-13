// Тиковый архив Bybit → бары 15m. Go-версия того же расчёта, что делает
// btcproc/ingest/bybit.py (parse_ticks + ticks_to_bars).
//
// Задача версии — не «переписать на Go», а дать ЧЕСТНОЕ сравнение: те же
// входные архивы, тот же выход до десятого знака, те же конвенции биржи.
// Все неочевидные правила перенесены из питоновского оригинала намеренно и
// помечены ссылками на него; отступление от любого из них сделало бы замер
// сравнением двух разных расчётов, а не двух реализаций одного.
//
// Отличие по устройству ровно одно, и оно и есть предмет замера: здесь
// однопроходная потоковая агрегация с постоянной памятью, а не материализация
// всех сделок в DataFrame.
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"syscall"
	"time"
)

const barMillis = 15 * 60 * 1000 // базовый ТФ 15m

// agg — накопитель одного бара. Всё, кроме закрытия, не зависит от порядка
// строк, поэтому считается на лету.
type agg struct {
	high, low      float64
	volume, quote  float64
	takerBuy       float64
	trades         int64
	closeTS        int64   // ключ последней сделки бара: время…
	closeFillOrder float64 // …и порядок исполнения внутри миллисекунды
	closePrice     float64
	seen           bool
}

// Восстановление порядка исполнения внутри одной миллисекунды.
// Оригинал: bybit.py, ticks_to_bars — ключ `price` для покупки и `-price` для
// продажи. Одна агрессивная заявка выедает стакан от лучшей цены к худшей, и
// без этого закрытие бара расходится с klines биржи на 26 баров из 1000.
func fillOrder(price float64, buy bool) float64 {
	if buy {
		return price
	}
	return -price
}

// later — тот же лексикографический порядок (ts, fillOrder), который в
// оригинале даёт двойная устойчивая сортировка.
func later(ts int64, fo float64, a *agg) bool {
	if ts != a.closeTS {
		return ts > a.closeTS
	}
	return fo > a.closeFillOrder
}

type fileResult struct {
	minBar, maxBar int64
	bars           map[int64]*agg
	ticks          int64
}

// timeUnitDivisor определяет, в чём метка времени, по порядку величины первого
// значения. Оригинал: bybit.py, _tick_timestamps. Bybit отдаёт миллисекунды на
// споте и секунды с дробной частью в части архивов деривативов; ошибка здесь
// уводит историю в 1970-й.
func timeUnitDivisor(first float64) float64 {
	switch {
	case first > 1e17:
		return 1e6 // ns → ms
	case first > 1e14:
		return 1e3 // us → ms
	case first > 1e11:
		return 1 // ms
	default:
		return 1.0 / 1000 // s → ms
	}
}

func processFile(path string) (*fileResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(bufio.NewReaderSize(f, 1<<20))
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	r := bufio.NewReaderSize(gz, 1<<20)
	res := &fileResult{bars: make(map[int64]*agg, 4096), minBar: math.MaxInt64, maxBar: math.MinInt64}

	var divisor float64
	first := true
	var fields [8][]byte

	for {
		line, err := r.ReadSlice('\n')
		if len(line) > 0 {
			// ReadSlice отдаёт срез внутреннего буфера — разбираем сразу,
			// ничего не копируя и не аллоцируя.
			n := splitCommas(line, fields[:])
			if n >= 5 {
				ok := parseRow(fields[:n], &divisor, first, res)
				if ok {
					first = false
				}
			}
		}
		if err != nil {
			break
		}
	}
	return res, nil
}

// splitCommas режет строку по запятым без аллокаций; хвостовые \n и \r
// снимаются с последнего поля.
func splitCommas(line []byte, out [][]byte) int {
	n, start := 0, 0
	for i := 0; i <= len(line) && n < len(out); i++ {
		if i == len(line) || line[i] == ',' {
			f := line[start:i]
			for len(f) > 0 && (f[len(f)-1] == '\n' || f[len(f)-1] == '\r') {
				f = f[:len(f)-1]
			}
			out[n] = f
			n++
			start = i + 1
		}
	}
	return n
}

func parseRow(f [][]byte, divisor *float64, first bool, res *fileResult) bool {
	// Колонки позиционные: id, timestamp, price, volume, side[, rpi].
	// Заголовку в этих архивах доверять нельзя (в месячных он перечисляет
	// пять имён при шести колонках данных) — оригинал по той же причине
	// задаёт имена по числу колонок, а строка заголовка отсеивается здесь
	// сама: её поля не парсятся как числа.
	tsRaw, err := strconv.ParseFloat(string(f[1]), 64)
	if err != nil {
		return false
	}
	price, err := strconv.ParseFloat(string(f[2]), 64)
	if err != nil {
		return false
	}
	volume, err := strconv.ParseFloat(string(f[3]), 64)
	if err != nil {
		return false
	}
	if first {
		*divisor = timeUnitDivisor(tsRaw)
	}
	ts := int64(tsRaw / *divisor)

	buy := len(f[4]) == 3 && (f[4][0] == 'b' || f[4][0] == 'B')

	idx := ts / barMillis
	if ts < 0 {
		idx = (ts - barMillis + 1) / barMillis
	}
	b := res.bars[idx]
	if b == nil {
		b = &agg{high: math.Inf(-1), low: math.Inf(1), closeTS: math.MinInt64,
			closeFillOrder: math.Inf(-1)}
		res.bars[idx] = b
		if idx < res.minBar {
			res.minBar = idx
		}
		if idx > res.maxBar {
			res.maxBar = idx
		}
	}
	b.seen = true
	if price > b.high {
		b.high = price
	}
	if price < b.low {
		b.low = price
	}
	b.volume += volume
	b.quote += price * volume
	b.trades++
	if buy {
		b.takerBuy += volume
	}
	fo := fillOrder(price, buy)
	if later(ts, fo, b) {
		b.closeTS, b.closeFillOrder, b.closePrice = ts, fo, price
	}
	res.ticks++
	return true
}

type bar struct {
	ts                      int64
	open, high, low, close  float64
	volume, quote, takerBuy float64
	trades                  int64
}

// finalize — вторая половина ticks_to_bars: пустые бары, биржевая конвенция
// открытия и включение открытия в диапазон.
func finalize(res *fileResult, prevClose *float64) []bar {
	if res.maxBar < res.minBar {
		return nil
	}
	out := make([]bar, 0, res.maxBar-res.minBar+1)
	last := math.NaN()
	if prevClose != nil {
		last = *prevClose
	}
	for idx := res.minBar; idx <= res.maxBar; idx++ {
		b := res.bars[idx]
		row := bar{ts: idx * barMillis}
		if b != nil && b.seen {
			row.high, row.low, row.close = b.high, b.low, b.closePrice
			row.volume, row.quote, row.takerBuy, row.trades = b.volume, b.quote, b.takerBuy, b.trades
			last = b.closePrice
		} else {
			// Бары без единой сделки биржа всё равно публикует: цена держится
			// на последнем закрытии, объёмы нулевые. Дыры в сетке базового ТФ
			// ломают скользящие окна признаков сильнее, чем плоский бар.
			if math.IsNaN(last) {
				continue // до первой сделки ffill нечем — строки нет вовсе
			}
			row.high, row.low, row.close = math.NaN(), math.NaN(), last
		}

		// Открытие = закрытие предыдущего бара (проверено сверкой с klines
		// Bybit: 199 из 199). Для самого первого бара выборки предыдущего
		// нет — берём его собственное закрытие.
		if len(out) > 0 {
			row.open = out[len(out)-1].close
		} else if prevClose != nil {
			row.open = *prevClose
		} else {
			row.open = row.close
		}
		// Открытие входит в диапазон: без этого high оказывается ниже
		// открытия на разрывах.
		if math.IsNaN(row.high) || row.open > row.high {
			row.high = row.open
		}
		if math.IsNaN(row.low) || row.open < row.low {
			row.low = row.open
		}
		out = append(out, row)
	}
	return out
}

func main() {
	out := flag.String("out", "-", "куда писать бары (CSV); - это stdout")
	workers := flag.Int("workers", 1, "сколько архивов разбирать параллельно")
	flag.Parse()
	paths := flag.Args()
	if len(paths) == 0 {
		fmt.Fprintln(os.Stderr, "нужен хотя бы один архив")
		os.Exit(2)
	}

	start := time.Now()

	// Разбор архивов независим — параллелится по файлам. Сшивание идёт строго
	// по порядку: открытие первого бара архива берётся из закрытия последнего
	// бара предыдущего, и порядок здесь нарушать нельзя.
	results := make([]*fileResult, len(paths))
	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var failure error
	for i, p := range paths {
		wg.Add(1)
		go func(i int, p string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			r, err := processFile(p)
			if err != nil {
				mu.Lock()
				if failure == nil {
					failure = fmt.Errorf("%s: %w", filepath.Base(p), err)
				}
				mu.Unlock()
				return
			}
			results[i] = r
		}(i, p)
	}
	wg.Wait()
	if failure != nil {
		fmt.Fprintln(os.Stderr, failure)
		os.Exit(1)
	}

	w := os.Stdout
	if *out != "-" {
		f, err := os.Create(*out)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer f.Close()
		w = f
	}
	bw := bufio.NewWriterSize(w, 1<<20)
	fmt.Fprintln(bw, "ts,open,high,low,close,volume,quote_volume,trades,taker_buy_base")

	var ticks, rows int64
	var prev *float64
	for _, res := range results {
		if res == nil {
			continue
		}
		ticks += res.ticks
		for _, b := range finalize(res, prev) {
			fmt.Fprintf(bw, "%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%d,%.10f\n",
				b.ts, b.open, b.high, b.low, b.close, b.volume, b.quote, b.trades, b.takerBuy)
			rows++
			c := b.close
			prev = &c
		}
	}
	bw.Flush()

	elapsed := time.Since(start).Seconds()
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	rssMB := float64(ru.Maxrss) / 1e6 // darwin отдаёт байты
	if runtime.GOOS == "linux" {
		rssMB = float64(ru.Maxrss) / 1e3 // linux — килобайты
	}

	enc := json.NewEncoder(os.Stderr)
	_ = enc.Encode(map[string]any{
		"impl":        "go",
		"archives":    len(paths),
		"workers":     *workers,
		"ticks":       ticks,
		"bars":        rows,
		"total_sec":   math.Round(elapsed*1000) / 1000,
		"peak_rss_mb": math.Round(rssMB*10) / 10,
	})
}
