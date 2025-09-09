import time
import threading

class TokenBucket:
    def __init__(self, rate_per_minute=60, capacity=None):
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.capacity = capacity or rate_per_minute
        self.tokens = self.capacity
        self.last = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens=1):
        with self.lock:
            now = time.time()
            # refill
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait(self, tokens=1):
        while True:
            if self.acquire(tokens):
                return
            time.sleep(0.05)

# Single bucket for Discogs; if you use per-user OAuth, maintain per-user buckets
discogs_bucket = TokenBucket(rate_per_minute=60)

def limit_discogs(func):
    def wrapper(*args, **kwargs):
        discogs_bucket.wait(1)
        return func(*args, **kwargs)
    return wrapper
