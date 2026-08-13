"""
Авторизация админки.

Правила жёсткие по умолчанию:
  * логин и пароль только из .env, никаких значений «по умолчанию» —
    сервис не стартует с заглушками (см. AdminConfig.validate);
  * сравнение паролей через compare_digest — постоянное время, чтобы по
    задержке ответа нельзя было подбирать пароль посимвольно;
  * подписанная кука с TTL вместо серверных сессий — перезапуск процесса
    не разлогинивает, но и подделать её без ADMIN_SECRET_KEY нельзя;
  * блокировка по IP после N неудачных попыток;
  * необязательный IP-allowlist поверх всего этого.
"""
from __future__ import annotations

import ipaddress
import logging
import secrets
import time
from dataclasses import dataclass, field

from fastapi import Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from btcproc import config

logger = logging.getLogger(__name__)

COOKIE_NAME = "btcproc_session"
_SALT = "btcproc-admin-session"


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(config.admin.secret_key, salt=_SALT)


@dataclass
class Attempts:
    count: int = 0
    locked_until: float = 0.0


@dataclass
class LoginGuard:
    """Счётчик неудачных попыток по IP. В памяти: админка — один процесс."""

    by_ip: dict[str, Attempts] = field(default_factory=dict)

    def locked_for(self, ip: str) -> int:
        entry = self.by_ip.get(ip)
        if not entry:
            return 0
        remaining = int(entry.locked_until - time.time())
        return max(0, remaining)

    def register_failure(self, ip: str) -> None:
        entry = self.by_ip.setdefault(ip, Attempts())
        entry.count += 1
        if entry.count >= config.admin.max_login_attempts:
            entry.locked_until = time.time() + config.admin.lockout_seconds
            entry.count = 0
            logger.warning("IP %s заблокирован на %d сек после неудачных попыток входа",
                           ip, config.admin.lockout_seconds)

    def reset(self, ip: str) -> None:
        self.by_ip.pop(ip, None)


guard = LoginGuard()


def client_ip(request: Request) -> str:
    """
    Адрес клиента для allowlist и счётчика неудачных входов.

    X-Forwarded-For читается ТОЛЬКО при ADMIN_TRUST_PROXY=true. Без прокси
    заголовок ставит сам клиент, то есть атакующий выбирает себе «IP»: любой
    адрес из ADMIN_IP_ALLOWLIST открывает дверь, а новый фейковый адрес на
    каждую попытку сводит на нет brute-force lockout — блокировка не
    накапливается никогда. Штатная схема развёртывания (deploy/README.md) —
    админка смотрит наружу напрямую, поэтому дефолт false.
    """
    if config.admin.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def ip_allowed(ip: str) -> bool:
    allowlist = config.admin.ip_allowlist
    if not allowlist:
        return True
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for entry in allowlist:
        try:
            if "/" in entry:
                if address in ipaddress.ip_network(entry, strict=False):
                    return True
            elif address == ipaddress.ip_address(entry):
                return True
        except ValueError:
            logger.warning("Некорректная запись в ADMIN_IP_ALLOWLIST: %s", entry)
    return False


def check_credentials(user: str, password: str) -> bool:
    """Оба сравнения выполняются всегда — время ответа не зависит от того,
    ошибся пользователь в логине или в пароле."""
    user_ok = secrets.compare_digest(user.encode(), config.admin.user.encode())
    password_ok = secrets.compare_digest(password.encode(), config.admin.password.encode())
    return user_ok and password_ok


def issue_session(user: str) -> str:
    return _serializer().dumps({"user": user, "issued": int(time.time())})


def read_session(token: str | None) -> dict | None:
    if not token:
        return None
    try:
        return _serializer().loads(token, max_age=config.admin.session_ttl)
    except SignatureExpired:
        return None
    except BadSignature:
        logger.warning("Отклонена кука с неверной подписью")
        return None


def current_user(request: Request) -> str | None:
    session = read_session(request.cookies.get(COOKIE_NAME))
    return session.get("user") if session else None
