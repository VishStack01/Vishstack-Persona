import pytest

from app.util import parse_json, strip_pii


def test_parse_plain_fenced_think_and_embedded():
    assert parse_json('{"a":1}') == {"a": 1}
    assert parse_json('Here:\n```json\n{"a":2}\n```') == {"a": 2}
    assert parse_json('<think>{"wrong":true} reasoning</think>\n{"a":3}') == {"a": 3}
    assert parse_json('Sure. {"a":[1,2]} Done.') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_strip_pii():
    out = strip_pii("mail me at a.b@x.com or 98765 43210, cc @vish.stack")
    assert "@x.com" not in out and "98765" not in out and "@vish" not in out
    assert strip_pii("CIBIL 750 chahiye") == "CIBIL 750 chahiye"
