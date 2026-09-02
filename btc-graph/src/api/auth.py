"""
Аутентификация публичного API по ключу из окружения.

Ключи лежат в `API_KEYS` — список через запятую, каждый элемент либо голый
ключ, либо `метка:ключ`. Метка нужна только для логов: по ней видно, чей
интегратор шумит, не выписывая сам ключ в журнал.

Три решения, которые легко откатить назад по незнанию:

* **пустой `API_KEYS` — это 503, а не «пускать всех».** Ручка публичного API
  отдаёт сохранённые оценки наружу; конфигурация без ключей означает
  «не настроено», и молча превращать её в открытый доступ нельзя. Тем же
  правилом живёт `ENABLE_CONFIG_RELOAD` в routes.py;
* **окружение читается на каждом запросе, а не на импорте.** Смена ключа не
  должна требовать пересборки образа — достаточно `make reload`. Цена —
  один `os.environ.get` на запрос, что на фоне похода в PostgreSQL не видно;
* **сравнение через `hmac.compare_digest`.** Обычное `==` на строках
  выходит на первом несовпавшем байте, и время ответа начинает подсказывать
  подбирающему длину общего префикса.

Ключ передаётся заголовком `X-API-Key` либо `Authorization: Bearer <ключ>` —
второй вариант ради клиентов, у которых произвольные заголовки прописать
негде.
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

ENV_VAR = "API_KEYS"

# auto_error=False: ошибку формируем сами — с русским текстом и одинаковым
# кодом для «ключа нет» и «ключ не тот».
_header_scheme = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Ключ доступа к публичному API (см. API_KEYS в окружении)",
)


def parse_keys(raw: str) -> dict[str, str]:
    """
    `"reader:abc, xyz"` → `{"abc": "reader", "xyz": "key2"}`.

    Ключ без метки получает позиционное имя: в логе всё равно должно быть
    чем различать источники.
    """
    keys: dict[str, str] = {}
    for position, chunk in enumerate(raw.split(","), start=1):
        item = chunk.strip()
        if not item:
            continue
        label, separator, secret = item.partition(":")
        secret = secret.strip()
        if separator and secret:
            keys[secret] = label.strip() or f"key{position}"
        else:
            keys[item] = f"key{position}"
    return keys


def load_keys() -> dict[str, str]:
    """Действующие ключи из окружения. Пустой словарь = API не настроено."""
    return parse_keys(os.environ.get(ENV_VAR, ""))


def api_key_configured() -> bool:
    return bool(load_keys())


def _from_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


def require_api_key(
    header_key: str | None = Security(_header_scheme),
    authorization: str | None = Header(default=None, include_in_schema=False),
) -> str:
    """
    FastAPI-зависимость: пускает дальше и возвращает метку ключа.

    503 — ключи не настроены, 401 — ключа нет или он неизвестен. Разные коды
    именно для того, чтобы «забыли положить API_KEYS в .env» не выглядело как
    «клиент прислал не тот ключ».
    """
    keys = load_keys()
    if not keys:
        raise HTTPException(
            status_code=503,
            detail=(
                "Публичное API не настроено: в окружении пуст API_KEYS. "
                "Пропускать запросы без ключа система не станет."
            ),
        )

    presented = header_key or _from_bearer(authorization)
    if not presented:
        raise HTTPException(
            status_code=401,
            detail="Нужен ключ: заголовок X-API-Key или Authorization: Bearer <ключ>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    presented_bytes = presented.encode("utf-8")
    for secret, label in keys.items():
        if hmac.compare_digest(presented_bytes, secret.encode("utf-8")):
            return label

    # Сам ключ в лог не попадает даже частично: обрезок ключа — всё ещё ключ.
    logger.warning("Отклонён запрос с неизвестным ключом (длина %d)", len(presented))
    raise HTTPException(
        status_code=401,
        detail="Неизвестный ключ",
        headers={"WWW-Authenticate": "Bearer"},
    )
