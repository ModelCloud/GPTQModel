import secrets
import string


def get_random_string(length: int = 8) -> str:
    """Generate a random string of fixed length with lowercase English letters."""
    alphabet = string.ascii_lowercase
    return ''.join(secrets.choice(alphabet) for _ in range(length))
