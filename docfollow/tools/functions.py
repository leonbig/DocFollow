from functools import wraps
import time
import os
import json
import datetime

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {str(e)}")
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"All {max_retries} attempts failed. Last error: {str(e)}")
                        raise last_exception
            return None
        return wrapper
    return decorator


@retry_on_failure(max_retries=3)
def save_text_to_file(content, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Error saving file {filename}: {e}")
        raise


@retry_on_failure(max_retries=3)
def save_scores_to_file(scores, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving scores to {filename}: {e}")
        raise

def log_message(message: str, log_file: str = None):
    """Log a message with timestamp to console and optionally to a file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')