import math

import pytest

from pdf2latex.metrics import (
    latex_tokens,
    metric_exact_match,
    metric_normalized_edit_similarity,
    metric_normalized_exact_match,
    normalize_latex,
    strip_math_delims,
)


@pytest.mark.parametrize(
    "pred,target,expected",
    [
        ("\\frac{1}{2}", "\\frac{1}{2}", 1.0),
        ("\\frac{1}{2}", "\\frac{2}{3}", 0.0),
        ("", "", 1.0),
    ],
)
def test_metric_exact_match(pred, target, expected):
    assert metric_exact_match(pred, target) == expected


@pytest.mark.parametrize(
    "src,expected",
    [
        (r"\[ x + y \]", "x + y"),
        (r"\( x + y \)", "x + y"),
        (r"$$ x + y $$", "x + y"),
        (r"$ x + y $", "x + y"),
        (r"\[x$", r"\[x$"),
    ],
)
def test_strip_math_delims(src, expected):
    assert strip_math_delims(src) == expected


@pytest.mark.parametrize(
    "src,expected",
    [
        ("  $$  x \\quad +~ y   $$  ", "x + y"),
        (r"\alpha\,\beta", r"\alpha\beta"),
        ("x  +   y", "x + y"),
        ("", ""),
    ],
)
def test_normalize_latex(src, expected):
    assert normalize_latex(src) == expected


@pytest.mark.parametrize(
    "pred,target,expected",
    [
        ("x + y", "$$ x + y $$", 1.0),
        ("x  +   y", "x + y", 1.0),
        ("x - y", "x + y", 0.0),
    ],
)
def test_metric_normalized_exact_match(pred, target, expected):
    assert metric_normalized_exact_match(pred, target) == expected


@pytest.mark.parametrize(
    "src,expected",
    [
        (r"\frac{1}{2} + x^2", [r"\frac", "{", "1", "}", "{", "2", "}", "+", "x", "^", "2"]),
        (r"x_{i,j}", ["x", "_", "{", "i", ",", "j", "}"]),
        ("=", ["="]),
        ("   ", []),
    ],
)
def test_latex_tokens(src, expected):
    assert latex_tokens(src) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("x + y", "x + y", 1.0),
        ("x + y", "x - y", 1 - 1 / 3),
        ("$$ x $$", "x", 1.0),
        ("", "", 1.0),
        ("0", "", 0.0),
    ],
)
def test_metric_normalized_edit_similarity(gold, pred, expected):
    assert math.isclose(
        metric_normalized_edit_similarity(gold, pred),
        expected,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
