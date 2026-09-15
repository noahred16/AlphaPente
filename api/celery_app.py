from celery import Celery

from api.config import settings

celery_app = Celery(
    "alphapente",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# `celery -A api.celery_app worker` only ever imports this module, so tasks
# never get registered just by existing in api/tasks/ - this import runs
# their @celery_app.task decorators. (autodiscover_tasks looks like the more
# "Celery-idiomatic" way to do this, but it defers the actual import until
# app finalization, which a plain `celery worker` startup doesn't trigger
# early enough - confirmed empirically: the worker's startup banner still
# showed an empty [tasks] list with it. A direct import is deterministic.)
# Must come after `celery_app` is assigned above: api.tasks.book imports it
# back (`from api.celery_app import celery_app`), so this order avoids a
# circular-import failure.
import api.tasks  # noqa: E402,F401
