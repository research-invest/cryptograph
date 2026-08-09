from __future__ import annotations

import time

import pytest

from btcproc import config
from btcproc.admin import auth


def _admin(**overrides) -> config.AdminConfig:
    base = {
        "user": "operator",
        "password": "very-long-password-42",
        "secret_key": "k" * 64,
        "session_ttl": 60,
        "max_login_attempts": 3,
        "lockout_seconds": 30,
        "ip_allowlist": [],
    }
    base.update(overrides)
    return config.AdminConfig(**base)


@pytest.fixture(autouse=True)
def strict_admin(monkeypatch):
    """AdminConfig заморожен, поэтому подменяем объект целиком."""
    monkeypatch.setattr(config, "admin", _admin())
    auth.guard.by_ip.clear()


def test_placeholder_credentials_are_rejected():
    """Сервис не должен подниматься с шаблонными значениями из .env.example."""
    weak = config.AdminConfig(
        user="admin",
        password="ЗАМЕНИ_МЕНЯ_длинным_паролем",
        secret_key="ЗАМЕНИ_МЕНЯ_на_вывод_openssl_rand_hex_32",
    )
    with pytest.raises(RuntimeError) as exc:
        weak.validate()
    assert "ADMIN_PASSWORD" in str(exc.value)


def test_short_password_and_key_are_rejected():
    with pytest.raises(RuntimeError):
        config.AdminConfig(user="op", password="short", secret_key="k" * 64).validate()
    with pytest.raises(RuntimeError):
        config.AdminConfig(user="op", password="long-enough-pass", secret_key="k" * 8).validate()


def test_valid_credentials_pass_validation():
    config.AdminConfig(
        user="op", password="long-enough-password", secret_key="k" * 64
    ).validate()


def test_credentials_check():
    assert auth.check_credentials("operator", "very-long-password-42")
    assert not auth.check_credentials("operator", "wrong")
    assert not auth.check_credentials("someone", "very-long-password-42")


def test_session_roundtrip_and_expiry(monkeypatch):
    token = auth.issue_session("operator")
    assert auth.read_session(token)["user"] == "operator"

    # Просроченная кука не принимается.
    monkeypatch.setattr(config, "admin", _admin(session_ttl=0))
    time.sleep(1.1)
    assert auth.read_session(token) is None


def test_tampered_cookie_rejected():
    token = auth.issue_session("operator")
    assert auth.read_session(token[:-3] + "abc") is None
    assert auth.read_session("совсем не токен") is None


def test_login_guard_locks_after_failures():
    for _ in range(3):
        assert auth.guard.locked_for("10.0.0.1") == 0
        auth.guard.register_failure("10.0.0.1")
    assert auth.guard.locked_for("10.0.0.1") > 0
    # Блокировка адресная: другой IP не страдает.
    assert auth.guard.locked_for("10.0.0.2") == 0

    auth.guard.reset("10.0.0.1")
    assert auth.guard.locked_for("10.0.0.1") == 0


def test_ip_allowlist(monkeypatch):
    assert auth.ip_allowed("203.0.113.7")

    monkeypatch.setattr(config, "admin", _admin(ip_allowlist=["127.0.0.1", "10.0.0.0/8"]))
    assert auth.ip_allowed("127.0.0.1")
    assert auth.ip_allowed("10.4.5.6")
    assert not auth.ip_allowed("203.0.113.7")
    assert not auth.ip_allowed("не-адрес")


# ── Доверие к X-Forwarded-For (B6) ──────────────────────────────────────────
#
# Без прокси заголовок ставит сам клиент. Раньше он читался безусловно, и это
# давало атакующему право выбрать себе адрес: подставить разрешённый в
# ADMIN_IP_ALLOWLIST и обнулять brute-force lockout новым фейковым IP на
# каждую попытку — блокировка не накапливалась никогда.


class _FakeClient:
    host = "203.0.113.9"


class _FakeRequest:
    def __init__(self, forwarded: str | None = None):
        self.headers = {"x-forwarded-for": forwarded} if forwarded else {}
        self.client = _FakeClient()


def test_forwarded_header_is_ignored_without_proxy(monkeypatch):
    monkeypatch.setattr(config, "admin", _admin(trust_proxy=False))

    ip = auth.client_ip(_FakeRequest("10.0.0.1, 198.51.100.7"))

    assert ip == "203.0.113.9", "подделанный заголовок не должен подменять адрес"


def test_forwarded_header_is_used_behind_trusted_proxy(monkeypatch):
    monkeypatch.setattr(config, "admin", _admin(trust_proxy=True))

    ip = auth.client_ip(_FakeRequest("10.0.0.1, 198.51.100.7"))

    assert ip == "10.0.0.1", "за доверенным прокси берём первый адрес цепочки"


def test_missing_header_falls_back_to_socket(monkeypatch):
    monkeypatch.setattr(config, "admin", _admin(trust_proxy=True))

    assert auth.client_ip(_FakeRequest()) == "203.0.113.9"


def test_spoofed_header_cannot_bypass_allowlist(monkeypatch):
    """Сценарий обхода целиком: allowlist + подделанный заголовок."""
    monkeypatch.setattr(
        config, "admin", _admin(trust_proxy=False, ip_allowlist=["10.0.0.1"])
    )

    assert auth.ip_allowed(auth.client_ip(_FakeRequest("10.0.0.1"))) is False
