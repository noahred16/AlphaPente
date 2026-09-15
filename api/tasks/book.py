from api.celery_app import celery_app


@celery_app.task(name="book.evaluate")
def evaluate_position(uniq_hash: str, target_visits: int | None = None) -> None:
    """Run MCTS evaluation on a position and update its book entry."""
    # TODO: implement MCTS evaluation + book_db (RocksDict) update
    raise NotImplementedError
