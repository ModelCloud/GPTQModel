import random
import string


def get_random_string(length: int = 8) -> str:
    """Generate a random string of fixed length with lowercase English letters."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))
