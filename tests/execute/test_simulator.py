from __future__ import annotations

import json
from pathlib import Path

from execute import ExecutionStatus, SimulatedHardware


def _tools() -> list[dict]:
    path = (
        Path(__file__).resolve().parents[3]
        / "stc/outputs/models/route_v1_best_step1250_hf/tools_openai.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_updates_digital_twin():
    hardware = SimulatedHardware(_tools())
    result = hardware.execute(
        {"name": "set_temperature", "arguments": {"value": 24, "zone": "driver"}}
    )
    assert result.status is ExecutionStatus.SUCCESS
    state = hardware.snapshot()
    assert state["climate"]["temperature"]["driver"] == 24
    assert state["climate"]["power"] == "on"
    assert state["action_count"] == 1


def test_required_parameter_and_busy_are_distinct():
    hardware = SimulatedHardware(_tools())
    missing = hardware.execute({"name": "control_trunk", "arguments": {}})
    assert missing.status is ExecutionStatus.MISSING_REQUIRED
    assert missing.missing == ("action",)

    hardware.set_busy(True)
    busy = hardware.execute(
        {"name": "control_trunk", "arguments": {"action": "open"}}
    )
    assert busy.status is ExecutionStatus.BUSY
    assert hardware.snapshot()["access"]["trunk"] == "closed"


def test_semantic_required_fields_are_enforced_for_optional_schema():
    hardware = SimulatedHardware(_tools())
    result = hardware.execute(
        {"name": "set_temperature", "arguments": {"zone": "driver"}}
    )
    assert result.status is ExecutionStatus.MISSING_REQUIRED
    assert result.error == "missing_actionable_field"
    assert result.details == {"one_of": ["value", "delta"]}
    assert hardware.snapshot()["action_count"] == 0


def test_bad_argument_shape_and_unknown_fields_are_rejected_without_crashing():
    hardware = SimulatedHardware(_tools())
    malformed = hardware.execute(
        {"name": "set_temperature", "arguments": "value=24"}
    )
    assert malformed.status is ExecutionStatus.REJECTED
    assert malformed.error == "invalid_arguments_type"

    unexpected = hardware.execute(
        {"name": "set_temperature", "arguments": {"value": 24, "hack": True}}
    )
    assert unexpected.status is ExecutionStatus.REJECTED
    assert unexpected.error == "unknown_arguments"
    assert unexpected.details["fields"] == ["hack"]
    assert hardware.snapshot()["action_count"] == 0


def test_out_of_range_temperature_does_not_mutate_state():
    hardware = SimulatedHardware(_tools())
    before = hardware.snapshot()
    result = hardware.execute(
        {"name": "set_temperature", "arguments": {"value": 12, "zone": "driver"}}
    )
    assert result.status is ExecutionStatus.REJECTED
    assert result.error == "invalid_arguments"
    assert result.details["minimum"] == 16
    assert hardware.snapshot() == before


def test_fahrenheit_is_accepted_only_when_explicit_and_converted_for_hardware():
    hardware = SimulatedHardware(_tools())
    result = hardware.execute(
        {
            "name": "set_temperature",
            "arguments": {"value": 72, "unit": "fahrenheit", "zone": "driver"},
        }
    )
    assert result.status is ExecutionStatus.SUCCESS
    assert hardware.snapshot()["climate"]["temperature"]["driver"] == 22

    implicit = hardware.execute(
        {"name": "set_temperature", "arguments": {"value": 72, "zone": "driver"}}
    )
    assert implicit.status is ExecutionStatus.REJECTED


def test_opening_trunk_is_rejected_until_vehicle_is_stopped():
    hardware = SimulatedHardware(_tools())
    hardware.sync_vehicle_status(running=True, speed_kph=8)
    before = hardware.snapshot()
    rejected = hardware.execute(
        {"name": "control_trunk", "arguments": {"action": "open"}}
    )
    assert rejected.status is ExecutionStatus.REJECTED
    assert rejected.error == "vehicle_not_stopped"
    assert hardware.snapshot() == before

    hardware.sync_vehicle_status(running=False, speed_kph=0)
    success = hardware.execute(
        {"name": "control_trunk", "arguments": {"action": "open"}}
    )
    assert success.status is ExecutionStatus.SUCCESS
    assert hardware.snapshot()["access"]["trunk"] == "open"


def test_radio_conditional_fields_request_follow_up():
    hardware = SimulatedHardware(_tools())
    result = hardware.execute(
        {"name": "control_radio", "arguments": {"action": "tune"}}
    )
    assert result.status is ExecutionStatus.MISSING_REQUIRED
    assert result.missing == ("frequency_or_station_name",)
