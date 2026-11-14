"""Tests for the two-level (family) blending used by the final submission."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import ensemble


def test_load_families_reads_a_json_object(tmp_path):
    path = tmp_path / "families.json"
    path.write_text(json.dumps({"CB": ["E5b"], "RF": ["RF1", "RF2"]}), encoding="utf-8")

    families = ensemble._load_families(path)
    assert families == {"CB": ["E5b"], "RF": ["RF1", "RF2"]}


@pytest.mark.parametrize("payload", ["{}", "[]", '{"CB": []}'])
def test_load_families_rejects_empty_payload(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        ensemble._load_families(path)


def test_family_blend_weights_families_equally_not_members():
    """One member vs three members must still count as one vote each."""
    oof_map = {
        "a": np.array([[0.8, 0.1, 0.1]]),
        "b1": np.array([[0.2, 0.7, 0.1]]),
        "b2": np.array([[0.2, 0.7, 0.1]]),
        "b3": np.array([[0.2, 0.7, 0.1]]),
    }
    test_map = {name: value.copy() for name, value in oof_map.items()}
    families = {"A": ["a"], "B": ["b1", "b2", "b3"]}

    oof, test = ensemble._family_blend(families, oof_map, test_map)

    # family A = [0.8, 0.1, 0.1], family B = [0.2, 0.7, 0.1] -> equal-weight mean
    assert np.allclose(oof, [[0.5, 0.4, 0.1]])
    assert np.allclose(test, [[0.5, 0.4, 0.1]])
    # averaging the four members directly would give a different answer
    naive = np.mean([oof_map[n] for n in ("a", "b1", "b2", "b3")], axis=0)
    assert not np.allclose(oof, naive)


def test_family_blend_matches_manual_two_level_average():
    rng = np.random.default_rng(0)
    oof_map = {name: rng.random((5, 3)) for name in ("x1", "x2", "y1")}
    test_map = {name: rng.random((4, 3)) for name in ("x1", "x2", "y1")}
    families = {"X": ["x1", "x2"], "Y": ["y1"]}

    oof, _ = ensemble._family_blend(families, oof_map, test_map)

    expected_x = (oof_map["x1"] + oof_map["x2"]) / 2
    expected = (expected_x + oof_map["y1"]) / 2
    assert np.allclose(oof, expected)
