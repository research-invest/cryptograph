"""
Точка входа. Запуск: uvicorn src.main:app --reload
"""
from dotenv import load_dotenv

# Читаем .env до импорта модулей, которые смотрят в окружение на импорте
# (src.db.connection создаёт engine, src.agent.pipeline читает USE_*).
# В Docker переменные приходят из env_file/environment и уже выставлены —
# load_dotenv их не перезаписывает.
load_dotenv()

import uvicorn  # noqa: E402
from src.api.routes import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
