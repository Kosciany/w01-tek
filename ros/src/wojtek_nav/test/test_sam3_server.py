"""The SAM3 server's pure parts: no torch, no model, no network."""

import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sam3 = load_script("sam3_server")


def test_importing_the_server_needs_no_torch():
    # The model's imports live inside Sam3(); the brain's tests and the
    # ROS container never pull torch in.
    assert "torch" not in vars(sam3)


def test_interior_point_of_a_box_is_its_centre():
    m = np.zeros((100, 200), bool)
    m[20:61, 50:151] = True  # rows 20..60, cols 50..150
    u, v = sam3.interior_point(m)
    assert u == pytest.approx(100.5 / 200, abs=0.01) and v == pytest.approx(40.5 / 100, abs=0.01)


def test_interior_point_stays_inside_a_ring():
    """A mask whose centroid is a hole: the point must still be on it."""
    yy, xx = np.mgrid[:200, :200]
    r = np.hypot(yy - 100, xx - 100)
    m = (r > 50) & (r < 80)
    u, v = sam3.interior_point(m)
    assert m[int(v * 200), int(u * 200)]


def test_interior_point_moves_away_from_the_frame_edge():
    m = np.zeros((100, 100), bool)
    m[:, 90:] = True  # an object cut by the right edge
    u, _ = sam3.interior_point(m)
    assert 0.9 < u < 0.99


@pytest.mark.parametrize("mask", [np.zeros((10, 10), bool), np.ones(5, bool)])
def test_no_point_for_an_empty_or_malformed_mask(mask):
    assert sam3.interior_point(mask) is None


def test_detections_are_normalised_and_sorted_and_drop_empty_masks():
    masks = np.zeros((3, 100, 200), bool)
    masks[0, 10:30, 10:30] = True
    masks[2, 50:90, 100:180] = True  # masks[1] stays empty
    boxes = np.array([[10, 10, 30, 30], [0, 0, 1, 1], [100, 50, 180, 90]], float)
    dets = sam3.to_detections(masks, boxes, np.array([0.6, 0.99, 0.8]), 200, 100)
    assert [d["score"] for d in dets] == [0.8, 0.6]
    assert dets[0]["box"] == [0.5, 0.5, 0.9, 0.9]
    assert dets[0]["area"] == pytest.approx(80 * 40 / 20000)
    json.dumps(dets)  # plain Python numbers only


def _body(**kw):
    req = {"image": base64.b64encode(b"\xff\xd8jpeg").decode(), "prompt": "purple pillar"}
    req.update(kw)
    return json.dumps(req).encode()


def test_parse_request():
    image, prompt, threshold = sam3.parse_request(_body(prompt="  red chair ", threshold=0.4))
    assert image == b"\xff\xd8jpeg" and prompt == "red chair" and threshold == 0.4
    assert sam3.parse_request(_body())[2] == 0.5


@pytest.mark.parametrize("body", [
    b"not json", b"[1]", _body(prompt=""), _body(prompt=3), _body(image=""),
    _body(image="%%%"), _body(threshold=2), _body(threshold=True), _body(threshold="0.5"),
])
def test_bad_requests_are_refused(body):
    with pytest.raises(ValueError):
        sam3.parse_request(body)
