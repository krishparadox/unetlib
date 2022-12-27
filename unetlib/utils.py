import math


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def is_power_two(n):
    return math.log2(n).is_integer()


def is_divisible(num, denom):
    return (num % denom) == 0


def cast_to_tuple(val, length=None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output
