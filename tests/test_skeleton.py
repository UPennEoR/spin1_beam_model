# -*- coding: utf-8 -*-

import pytest

from spin1_beam_model.skeleton import fib

__author__ = "Steven Murray"
__copyright__ = "Steven Murray"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
