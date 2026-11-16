"""The VLM brain's policy on a desk: answers in, actions out, no model."""

import math

import pytest

from wojtek_nav.vlm_brain import (
    CONCEPT_SCHEMA,
    Explorer,
    SCHEMA,
    answer_from_detections,
    chat_url,
    concept_prompt,
    detect_url,
    normalised_point,
    parse_answer,
    parse_concept,
    pixel_ray,
    ray_heading,
    rotate,
    step_along,
    task_prompt,
    verify_prompt,
)


@pytest.mark.parametrize("base", [
    "http://box:11434", "http://box:11434/", "http://box:11434/v1", "http://box:11434/v1/",
])
def test_chat_url_takes_a_base_url_with_or_without_v1(base):
    assert chat_url(base) == "http://box:11434/v1/chat/completions"


def test_schema_is_the_whole_vocabulary():
    assert SCHEMA["properties"]["type"]["enum"] == ["goal", "turn", "not_visible", "done"]
    assert SCHEMA["additionalProperties"] is False


def test_prompts_carry_the_instruction_verbatim():
    assert "podejdź do fioletowego słupa" in task_prompt("podejdź do fioletowego słupa")
    v = verify_prompt("podejdź do niskiej skrzynki", "skrzynka")
    assert "podejdź do niskiej skrzynki" in v and "skrzynka" in v
    assert "size" in v  # the property that a big crate failed on


def test_parse_tolerates_wrapping_and_rejects_prose():
    assert parse_answer('```json\n{"type":"done"}\n```')["type"] == "done"
    assert parse_answer("I see a crate at the left")["type"] == "unparsable"


def test_normalised_point_bounds():
    assert normalised_point({"point_2d": [500, 250]}) == (0.5, 0.25)
    assert normalised_point({"point_2d": [1001, 0]}) is None
    assert normalised_point({"point_2d": [1]}) is None
    assert normalised_point({}) is None
    # Digit strings are still a usable point.
    assert normalised_point({"point_2d": ["500", "250"]}) == (0.5, 0.25)


@pytest.mark.parametrize("bad", [
    [500], [], [1, 2, 3], "500,250", 500, {"x": 500, "y": 250}, [None, 5], [[500], [250]],
    float("nan"), [float("nan"), 0],
])
def test_normalised_point_rejects_malformed_shapes(bad):
    # The node draws and acts on this value only; the raw point_2d never
    # reaches arithmetic.
    assert normalised_point({"type": "goal", "point_2d": bad}) is None


def test_goal_is_verified_before_it_moves():
    e = Explorer()
    assert e.on_answer({"type": "goal", "point_2d": [600, 250], "label": "słup"}) == ("verify", "słup")
    assert e.on_verified(True) == ("goal", (0.6, 0.25))


def test_unverified_goal_becomes_a_search():
    e = Explorer(turn_deg=45)
    e.on_answer({"type": "goal", "point_2d": [600, 250], "label": "x"})
    assert e.on_verified(False) == ("search", 45.0)


def test_not_visible_turns_and_a_full_circle_explores():
    e = Explorer(turn_deg=90)
    for _ in range(4):
        assert e.on_answer({"type": "not_visible"}) == ("search", 90.0)
    assert e.on_answer({"type": "not_visible"}) == ("explore", 1.0)
    assert e.on_answer({"type": "not_visible"}) == ("search", 90.0)  # the circle restarts


def test_models_own_turn_is_honoured():
    e = Explorer()
    assert e.on_answer({"type": "turn", "deg": -60}) == ("search", -60.0)


def test_done_is_the_executives_call_not_the_models():
    e = Explorer(done_within_m=1.1)
    assert e.on_answer({"type": "done"}) == ("search", 45.0)  # not trusted
    e.on_answer({"type": "goal", "point_2d": [500, 500]})
    e.on_verified(True)
    assert e.on_goal_result("reached", distance_m=2.4) == ("look",)
    assert e.on_goal_result("reached", distance_m=0.85) == ("done",)


def _pointed(e, point=(500, 500)):
    """The model points, the check confirms: the explorer holds a pixel goal."""
    e.on_answer({"type": "goal", "point_2d": list(point)})
    return e.on_verified(True)


def test_far_target_is_approached_along_the_pixels_bearing():
    e = Explorer(approach_m=1.0)
    _pointed(e, (468, 244))
    # The pixel travels with the approach: the node walks its bearing.
    assert e.on_goal_result("no_depth") == ("approach", 1.0, (0.468, 0.244), 0.0)
    assert e.on_move_result("reached", moved_m=0.95) == ("look",)


def test_blocked_approach_looks_again_and_detours_alternating_sides():
    e = Explorer(detour_deg=40)
    detours, after = [], []
    for _ in range(3):
        _pointed(e)
        detours.append(e.on_goal_result("no_depth")[3])
        after.append(e.on_move_result("blocked", moved_m=0.6))
    assert detours == [0.0, 40.0, -40.0]
    # The straight walk looks again; a detoured one turns back onto the
    # bearing first.
    assert after == [("look",), ("search", -40.0), ("search", 40.0)]
    assert e.turn_done() == ("look",)
    _pointed(e)
    e.on_goal_result("no_depth")
    e.on_move_result("reached", moved_m=1.0)
    _pointed(e)
    assert e.on_goal_result("no_depth")[3] == 0.0  # a clear walk restores the bearing


def _detoured(e):
    """A blocked straight approach, then the next approach goes 40 deg off
    the bearing."""
    _pointed(e)
    e.on_goal_result("no_depth")
    assert e.on_move_result("blocked", moved_m=0.6) == ("look",)
    _pointed(e)
    assert e.on_goal_result("no_depth")[3] == 40.0


def test_reached_detour_turns_back_onto_the_bearing_before_looking():
    """After the detour the body points 40 deg off the target, out of the
    picture: turn back first, or the look says not_visible."""
    e = Explorer(detour_deg=40)
    _detoured(e)
    assert e.on_move_result("reached", moved_m=1.0) == ("search", -40.0)
    assert e.turn_done() == ("look",)
    _pointed(e)
    assert e.on_goal_result("no_depth")[3] == 0.0


def test_turning_back_after_a_detour_does_not_count_towards_the_circle():
    e = Explorer(turn_deg=90, detour_deg=40, max_steps=100)
    _detoured(e)
    e.on_move_result("reached", moved_m=1.0)
    assert e.turned_deg == 0.0
    for _ in range(4):
        assert e.on_answer({"type": "not_visible"}) == ("search", 90.0)
    assert e.on_answer({"type": "not_visible"}) == ("explore", 1.0)


def test_explore_never_turns_back():
    e = Explorer(turn_deg=90, detour_deg=40, max_steps=100)
    _detoured(e)
    e.on_move_result("blocked", moved_m=0.3)  # the detour flips to -40 and waits
    for _ in range(5):
        e.on_answer({"type": "not_visible"})  # four turns, then the explore
    assert e.on_move_result("reached", moved_m=1.0) == ("look",)


def test_stalled_detour_still_gives_up():
    e = Explorer(detour_deg=40, min_progress_m=0.25, max_stalls=1, max_steps=100)
    _detoured(e)
    assert e.on_move_result("reached", moved_m=0.02)[0] == "gave_up"


def test_blocked_explore_still_turns():
    e = Explorer(turn_deg=90)
    for _ in range(5):
        e.on_answer({"type": "not_visible"})  # four turns, then the explore
    assert e.on_move_result("blocked", moved_m=0.5) == ("search", 90.0)


def test_blocked_near_the_target_is_done():
    e = Explorer(done_within_m=1.1)
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=0.95) == ("done",)
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.8) == ("look",)


def test_target_that_never_enters_depth_range_gives_up_with_a_reason():
    e = Explorer(max_blind=3, max_steps=100)
    for _ in range(3):
        _pointed(e)
        assert e.on_goal_result("no_depth")[0] == "approach"
        assert e.on_move_result("reached", moved_m=1.0) == ("look",)
    _pointed(e)
    kind, why = e.on_goal_result("no_depth")
    assert kind == "gave_up" and "depth range" in why


def test_resolved_depth_resets_the_blind_count():
    e = Explorer(max_blind=2, max_steps=100)
    for _ in range(2):
        _pointed(e)
        e.on_goal_result("no_depth")
        e.on_move_result("reached", moved_m=1.0)
    _pointed(e)
    assert e.on_goal_result("reached", distance_m=2.5) == ("look",)
    _pointed(e)
    assert e.on_goal_result("no_depth")[0] == "approach"


def test_moves_that_go_nowhere_give_up_with_a_reason():
    e = Explorer(min_progress_m=0.25, max_stalls=3, max_steps=100)
    for _ in range(2):
        _pointed(e)
        e.on_goal_result("no_depth")
        assert e.on_move_result("reached", moved_m=0.02) == ("look",)
    _pointed(e)
    e.on_goal_result("no_depth")
    kind, why = e.on_move_result("reached", moved_m=0.02)
    assert kind == "gave_up" and "no progress" in why


def test_a_real_move_clears_the_stall_count():
    e = Explorer(min_progress_m=0.25, max_stalls=2, max_steps=100)
    for moved in (0.1, 0.8, 0.1):
        _pointed(e)
        e.on_goal_result("no_depth")
        assert e.on_move_result("reached", moved_m=moved) == ("look",)


def test_unknown_progress_is_not_a_stall():
    e = Explorer(max_stalls=1)
    _pointed(e)
    e.on_goal_result("no_depth")
    assert e.on_move_result("reached") == ("look",)


def test_pixel_ray_through_the_principal_point_is_the_optical_axis():
    k = [600.0, 0, 424.0, 0, 600.0, 240.0, 0, 0, 1]
    assert pixel_ray(424.0, 240.0, k) == (0.0, 0.0, 1.0)
    assert pixel_ray(1024.0, 240.0, k) == (1.0, 0.0, 1.0)


# Optical frame (x right, y down, z forward) in a body frame (x forward,
# y left, z up) looking straight ahead: the robot's camera, 0 deg pitch.
OPTICAL_IN_BODY = (-0.5, 0.5, -0.5, 0.5)


def _yaw_q(yaw, q):
    """q premultiplied by a yaw about z: the body turned by `yaw` in odom."""
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    x, y, z, w = q
    return (c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z)


def test_rotate_is_a_rotation():
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # 90 deg about z
    assert rotate(q, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))


@pytest.mark.parametrize("robot_yaw", [0.0, 0.7, -2.0])
def test_ray_heading_follows_the_pixel_and_the_body(robot_yaw):
    q = _yaw_q(robot_yaw, OPTICAL_IN_BODY)
    k = [600.0, 0, 424.0, 0, 600.0, 240.0, 0, 0, 1]
    assert ray_heading(q, pixel_ray(424.0, 240.0, k)) == pytest.approx(robot_yaw)
    # A pixel right of centre is a bearing to the right (negative yaw).
    right = ray_heading(q, pixel_ray(1024.0, 240.0, k))
    assert right == pytest.approx(robot_yaw - math.pi / 4)


def test_ray_heading_ignores_the_cameras_pitch():
    # base_link -> camera_color_optical_frame as the sim's TF has it: the
    # camera looks 15 deg down.
    q = (0.561, -0.561, 0.430, -0.430)
    k = [600.0, 0, 424.0, 0, 600.0, 240.0, 0, 0, 1]
    for v in (40.0, 240.0, 460.0):  # above, on and below the optical axis
        assert ray_heading(q, pixel_ray(424.0, v, k)) == pytest.approx(0.0, abs=1e-3)
    assert ray_heading(q, pixel_ray(1024.0, 240.0, k)) < -0.5


def test_step_along():
    assert step_along((1.0, 2.0), math.pi / 2, 1.5) == pytest.approx((1.0, 3.5))


def test_blocked_goal_looks_again_then_detours_instead_of_pushing_again():
    e = Explorer(detour_deg=40, max_steps=100)
    _pointed(e, (600, 300))
    assert e.on_goal_result("blocked", distance_m=2.7) == ("look",)  # re-resolve from here
    _pointed(e, (600, 300))
    # Round the obstacle along the pixel's bearing, not a search turn that
    # leaves the target in the picture and the next goal on the same line.
    assert e.on_goal_result("blocked", distance_m=2.4) == ("approach", 1.0, (0.6, 0.3), 40.0)
    # The detour ends turned back onto the bearing.
    assert e.on_move_result("reached", moved_m=1.0) == ("search", -40.0)
    e.turn_done()
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=2.0) == ("look",)  # the count restarted


def test_repeated_blocked_goal_detours_alternate_sides():
    e = Explorer(detour_deg=40, max_steps=100)
    sides = []
    for dist in (3.0, 2.5, 2.0):  # progress each time: no stall
        _pointed(e)
        e.on_goal_result("blocked", distance_m=dist + 0.2)
        _pointed(e)
        sides.append(e.on_goal_result("blocked", distance_m=dist)[3])
        e.on_move_result("blocked", moved_m=0.6)
        e.turn_done()
    assert sides == [40.0, -40.0, 40.0]


def test_blocked_goals_that_get_no_nearer_give_up_with_a_reason():
    """The 2026-11-16 sim run: blocked 1.6 m from the pillar, every look
    pointing at it again, until the 600 s budget. Now it ends early."""
    e = Explorer(min_progress_m=0.25, max_stalls=3, max_steps=100)
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.78) == ("look",)
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.68)[0] == "approach"  # stall 1
    e.on_move_result("blocked", moved_m=0.4)
    e.turn_done()
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.74) == ("look",)  # stall 2
    _pointed(e)
    kind, why = e.on_goal_result("blocked", distance_m=1.63)  # stall 3
    assert kind == "gave_up" and "no progress" in why and "1.78" in why


def test_progress_towards_the_target_clears_the_goal_stall_count():
    e = Explorer(min_progress_m=0.25, max_stalls=2, max_steps=100)
    for first, second in ((2.7, 2.65), (2.3, 2.25), (1.9, 1.85)):
        _pointed(e)
        assert e.on_goal_result("blocked", distance_m=first) == ("look",)
        _pointed(e)
        assert e.on_goal_result("blocked", distance_m=second)[0] == "approach"
        e.on_move_result("reached", moved_m=1.0)
        e.turn_done()


def test_unknown_distance_is_not_a_goal_stall():
    e = Explorer(max_stalls=1, max_steps=100)
    for _ in range(4):
        _pointed(e)
        assert e.on_goal_result("blocked")[0] != "gave_up"


# -- the optional SAM3 detector ------------------------------------------


@pytest.mark.parametrize("base", ["http://box:8100", "http://box:8100/", "http://box:8100/detect"])
def test_detect_url_takes_a_base_url(base):
    assert detect_url(base) == "http://box:8100/detect"


def test_concept_prompt_carries_the_instruction_and_asks_for_english():
    p = concept_prompt("podejdź do fioletowego słupa")
    assert "podejdź do fioletowego słupa" in p and "English" in p
    assert CONCEPT_SCHEMA["required"] == ["detectable", "prompt"]


@pytest.mark.parametrize("answer,concept", [
    ({"detectable": True, "prompt": "purple pillar"}, "purple pillar"),
    ({"detectable": True, "prompt": "  \"red   chair\". "}, "red chair"),
    ({"detectable": False, "prompt": "corridor"}, None),
    ({"detectable": True, "prompt": ""}, None),
    ({"detectable": True, "prompt": "walk to the end of the long corridor and stop"}, None),
    ({"detectable": "true", "prompt": "pillar"}, None),  # a string is not a yes
    ({"detectable": True, "prompt": 5}, None),
    ({"type": "unparsable"}, None),
    ("pillar", None),
])
def test_parse_concept(answer, concept):
    assert parse_concept(answer) == concept


def test_most_confident_detection_becomes_a_goal_the_explorer_verifies():
    dets = [{"score": 0.62, "point": [0.2, 0.4]}, {"score": 0.91, "point": [0.55, 0.25]}]
    a = answer_from_detections(dets, "purple pillar", min_score=0.5)
    assert a == {"type": "goal", "point_2d": [550, 250], "label": "purple pillar", "score": 0.91}
    e = Explorer()
    assert e.on_answer(a) == ("verify", "purple pillar")  # still asked before it moves
    assert e.on_verified(True) == ("goal", (0.55, 0.25))


@pytest.mark.parametrize("dets", [
    [], None, "x", [{"score": 0.3, "point": [0.5, 0.5]}],
    [{"score": 0.9}], [{"score": 0.9, "point": [1.5, 0.5]}], [{"score": "high", "point": [0.5, 0.5]}],
    [{"score": float("nan"), "point": [0.5, 0.5]}], [{"score": 0.9, "point": [float("nan"), 0.5]}],
    [{"score": 0.9, "point": [0.5]}], [None],
])
def test_no_usable_detection_is_not_visible(dets):
    assert answer_from_detections(dets, "pillar", min_score=0.5) == {"type": "not_visible"}
    assert Explorer(turn_deg=45).on_answer({"type": "not_visible"}) == ("search", 45.0)


def test_refresh_picks_the_instance_nearest_the_verified_one():
    dets = [{"score": 0.97, "point": [0.9, 0.2]}, {"score": 0.80, "point": [0.45, 0.3]}]
    assert answer_from_detections(dets, "pillar", near=(0.5, 0.3))["point_2d"] == [450, 300]
    assert answer_from_detections(dets, "pillar")["point_2d"] == [900, 200]


def test_retarget_moves_the_goal_and_the_approach_to_the_fresh_pixel():
    e = Explorer(approach_m=1.0)
    _pointed(e, (500, 500))
    assert e.retarget((0.52, 0.48)) == ("goal", (0.52, 0.48))
    assert e.on_goal_result("no_depth") == ("approach", 1.0, (0.52, 0.48), 0.0)
