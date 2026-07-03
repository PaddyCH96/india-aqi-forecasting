import time
import functools
import logging

from lib.config import DB_URL

logger = logging.getLogger("india-aqi.utils")


def retry(max_attempts=3, delay=2, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        wait = delay * (backoff ** (attempt - 1))
                        logger.warning(
                            "Attempt %d/%d failed: %s. Retrying in %ds...",
                            attempt, max_attempts, e, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error("All %d attempts failed: %s", max_attempts, e)
            raise last_exception
        return wrapper
    return decorator


def validate_db_url(url: str = DB_URL) -> bool:
    if not url.startswith("postgresql://"):
        raise ValueError(
            f"Invalid DB_URL: must start with 'postgresql://', got '{url[:30]}...'"
        )
    return True
