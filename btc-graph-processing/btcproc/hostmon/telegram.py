"""
Отправка сообщений в Telegram — канал уведомлений монитора.

Тонкий модуль поверх `sendMessage` Bot API. Отдельно от `btcproc/notify/`
сознательно: там вебхуки о кандидатах (внешний контракт с версией схемы в теле,
дедупликация доставок, окно свежести), здесь — короткий текст оператору о
состоянии машины. Общего у них ровно ноль, кроме слова «уведомление».

Токен нигде не должен оказаться в логе. Он входит в URL, поэтому любое
сообщение об ошибке, полученное от httpx, прогоняется через `_hide_token`:
трейсбек с полным URL — это утёкший бот.
"""
from __future__ import annotations

import html
import logging

from btcproc import config

logger = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/sendMessage"
TIMEOUT = 10


class TelegramError(RuntimeError):
    """Не удалось отправить: сеть, неверный токен, недоступный чат."""


def send(text: str, *, token: str | None = None, chat_id: str | None = None) -> None:
    """
    Отправить сообщение. Бросает `TelegramError` — вызывающий решает, что делать
    с неудачей (монитор её журналирует и продолжает работать).

    Разметка — HTML: подставляемые значения обязаны проходить через `esc()`,
    иначе имя процесса со знаком `<` уронит разбор на стороне Telegram, и
    сообщение не дойдёт целиком.
    """
    import httpx

    token = token or config.alerts.bot_token
    chat_id = chat_id or config.alerts.chat_id
    if not token or not chat_id:
        raise TelegramError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы")

    try:
        response = httpx.post(
            API.format(token=token),
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=TIMEOUT,
        )
    except Exception as exc:      # noqa: BLE001 — любая сетевая беда равнозначна
        raise TelegramError(_hide_token(f"{type(exc).__name__}: {exc}", token)) from None

    if response.status_code != 200:
        # Тело ответа Telegram содержит описание причины («chat not found»,
        # «bot was blocked by the user») — оно и нужно оператору.
        detail = _hide_token(response.text[:300], token)
        raise TelegramError(f"HTTP {response.status_code}: {detail}")


def esc(value) -> str:
    """Экранирование для parse_mode=HTML."""
    return html.escape(str(value), quote=False)


def _hide_token(text: str, token: str) -> str:
    return text.replace(token, "…") if token else text
