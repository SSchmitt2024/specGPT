import time
import threading
from functools import wraps

def ttl_cache(maxsize=1000, ttl=3600):
    cache = {}
    lock = threading.Lock()
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args and kwargs into a hashable key
            try:
                # We handle unhashable dicts by converting them to frozensets of items
                def _freeze(obj):
                    if hasattr(obj, "model_dump"):
                        return _freeze(obj.model_dump())
                    if isinstance(obj, dict):
                        return frozenset((k, _freeze(v)) for k, v in obj.items())
                    elif isinstance(obj, list):
                        return tuple(_freeze(v) for v in obj)
                    return obj
                
                key = (_freeze(args), _freeze(kwargs))
            except Exception:
                # Fallback if arguments are completely unhashable
                return func(*args, **kwargs)
                
            with lock:
                if key in cache:
                    val, ts = cache[key]
                    if time.time() - ts < ttl:
                        return val
                    else:
                        del cache[key]
            
            val = func(*args, **kwargs)
            
            with lock:
                if len(cache) >= maxsize:
                    # just clear everything on overflow for simplicity
                    cache.clear()
                cache[key] = (val, time.time())
                
            return val
        return wrapper
    return decorator
