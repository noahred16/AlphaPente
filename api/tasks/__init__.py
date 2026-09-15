# Import each task module so celery_app.autodiscover_tasks(["api"]) - which
# imports this package - actually registers their @celery_app.task functions.
# Add new task modules to both places: here, and their own file below.
from api.tasks import book  # noqa: F401
