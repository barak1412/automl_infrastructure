import string
from random import choice


def random_str(length, charset=string.digits):
    """returns random string with a given length."""
    return "".join(choice(charset) for _ in range(0, length))

