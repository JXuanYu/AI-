import threading


# Define the timeout error
class TimeoutError(Exception):
    pass


# Decorator to set the timeout
def timeout(seconds):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            # Result holder
            result = [None]

            # Function to raise the timeout error
            def _raise_timeout():
                raise TimeoutError("Function timed out")

            # Timer to trigger the timeout
            timer = threading.Timer(seconds, _raise_timeout)
            timer.start()
            try:
                result[0] = fn(*args, **kwargs)
            finally:
                timer.cancel()

            return result[0]

        return wrapper

    return decorator
