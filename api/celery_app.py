from celery import Celery

from api.config import settings

celery_app = Celery(
    "alphapente",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
