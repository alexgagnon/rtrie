from rtrie.stats import get_median, get_mode
import pytest

@pytest.mark.parametrize("input,expected", [
  (None, None),
  ({}, None),
  ({1:1}, 1),
  ({1:3}, 1),
  ({1:1, 2:1, 3:1}, 2),
  ({1:2, 2:1}, 1),
  ({1:3, 2:1}, 1)
])
def test_get_median(input, expected):
    assert(get_median(input) == expected)

@pytest.mark.parametrize("input,expected", [
  (None, None),
  ({}, None),
  ({1:1}, [1]),
  ({1:3}, [1]),
  ({1:1, 2:1, 3:1}, [1, 2, 3]),
  ({1:2, 2:1}, [1]),
  ({1:3, 2:1}, [1])
])
def test_get_mode(input, expected):
    assert(get_mode(input) == expected)