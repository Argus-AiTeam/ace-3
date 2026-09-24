"""Independent rational oracle for the isolated Q24 proposal, not a model oracle."""

from fractions import Fraction


ARITHMETIC_ID = "ace3-residual-exact-grid-q24-v1"


class PrimitiveFault(ValueError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def fp16_value(word: int) -> Fraction:
    if type(word) is not int or not 0 <= word <= 0xFFFF:
        raise PrimitiveFault(1, "FP16 word must be an unsigned 16-bit integer")
    exponent, fraction = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise PrimitiveFault(1, "nonfinite FP16 operand")
    if exponent == 0:
        value = Fraction(fraction, 1 << 24)
    else:
        value = Fraction(1024 + fraction, 1024) * Fraction(2) ** (exponent - 15)
    return -value if word & 0x8000 else value


def validate_state(integer: int, negzero: int) -> None:
    if type(integer) is not int or not -(1 << 63) <= integer < (1 << 63):
        raise PrimitiveFault(1, "state must be a signed 64-bit integer")
    if type(negzero) is not int or negzero not in (0, 1):
        raise PrimitiveFault(2, "negative-zero tag must be integer 0 or 1")
    if integer != 0 and negzero != 0:
        raise PrimitiveFault(2, "nonzero state cannot carry a negative-zero tag")


def project(integer: int, negzero: int) -> int:
    validate_state(integer, negzero)
    if integer == 0:
        return negzero << 15
    target = Fraction(abs(integer), 1 << 24)
    if target >= 65520:
        raise PrimitiveFault(4, "FP16 view rounds to infinity")
    # Search representable magnitudes rather than duplicating RTL bit rounding.
    low, high = 0, 0x7BFF
    while low < high:
        middle = (low + high + 1) // 2
        if fp16_value(middle) <= target:
            low = middle
        else:
            high = middle - 1
    candidates = (low,) if low == 0x7BFF else (low, low + 1)
    nearest = min(candidates, key=lambda word: (abs(fp16_value(word) - target), word & 1))
    return nearest | (0x8000 if integer < 0 else 0)


def root(word: int) -> tuple[int, int]:
    scaled = fp16_value(word) * (1 << 24)
    if scaled.denominator != 1:
        raise ArithmeticError("finite binary16 operand is not on the Q24 grid")
    return scaled.numerator, int(word == 0x8000)


def add(integer: int, negzero: int, word: int) -> tuple[int, int, int]:
    increment, _ = root(word)
    validate_state(integer, negzero)
    result = integer + increment
    if not -(1 << 63) <= result < (1 << 63):
        raise PrimitiveFault(3, "signed 64-bit residual overflow")
    tag = int(result == 0 and integer == 0 and negzero == 1 and word == 0x8000)
    return result, tag, project(result, tag)
