"""
Уведомления о новых кандидатах: POST с JSON на внешний адрес.

Публичный API пакета — две функции:

    notify.dispatch(candidate_ids)   поставить уведомления в очередь
    notify.flush()                   дождаться разбора очереди перед выходом

Всё остальное — детали: правила в `rules`, формат тела в `payload`, очередь и
транспорт в `sender`, связка между ними — в `service`.

Модуль называется `service`, а не `dispatch`, намеренно: имя функции
`notify.dispatch` перекрыло бы одноимённый подмодуль в пространстве имён
пакета, и `from btcproc.notify import dispatch` возвращало бы то функцию, то
модуль в зависимости от порядка импортов.
"""
from btcproc.notify.service import dispatch, flush, send_one

__all__ = ["dispatch", "flush", "send_one"]
