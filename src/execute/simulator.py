"""Schema-driven simulated hardware used by the end-to-end voice demo.

The SLM checkpoint currently carries the car tool catalog. This simulator is
deliberately behind the execution-layer boundary: replacing the catalog and
state projection with robot hardware must not change the demo harness.
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

import jsonschema


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    MISSING_REQUIRED = "missing_required"
    BUSY = "busy"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    name: str
    status: ExecutionStatus
    arguments: dict[str, Any]
    missing: tuple[str, ...] = ()
    error: str | None = None
    details: dict[str, Any] | None = None
    state: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status is ExecutionStatus.SUCCESS

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "ok": self.ok,
            "arguments": self.arguments,
            "missing": list(self.missing),
            "error": self.error,
            "details": copy.deepcopy(self.details) if self.details else {},
        }


def _initial_state() -> dict[str, Any]:
    return {
        "connection": "online",
        "mode": "ready",
        "busy": False,
        "battery": 82,
        "vehicle": {"running": False, "speed_kph": 0.0},
        "action_count": 0,
        "last_action": None,
        "climate": {
            "power": "off",
            "ac": False,
            "temperature": {"driver": 22, "passenger": 22, "rear": 22},
            "fan": 0,
            "direction": ["auto"],
            "recirculation": "auto",
        },
        "lighting": {
            "ambient": {"state": "off", "color": "white", "brightness": 50},
            "cabin": {"state": "off", "brightness": 100},
            "fog": "off",
            "auto_high_beam": "off",
        },
        "media": {
            "volume": 30,
            "muted": False,
            "source": "bluetooth",
            "title": None,
            "playback": "stopped",
        },
        "comfort": {
            "seat_heating": {"driver": 0, "passenger": 0, "rear": 0},
            "seat_ventilation": {"driver": 0, "passenger": 0, "rear": 0},
            "massage": {"driver": "off", "passenger": "off", "rear": "off"},
        },
        "access": {
            "windows": {"driver": 0, "passenger": 0, "rear_left": 0, "rear_right": 0},
            "sunroof": 0,
            "trunk": "closed",
            "mirrors": "unfolded",
        },
        "drive": {"mode": "normal", "adas": {}, "regen": "normal"},
        "connectivity": {"wifi": "off", "bluetooth": "on", "device": None},
        "other": {},
        "events": [],
    }


def _zones(value: str | None) -> list[str]:
    if value in {None, "all", "front"}:
        return ["driver", "passenger"] if value == "front" else ["driver", "passenger", "rear"]
    if value in {"rear", "rear_left", "rear_right"}:
        return ["rear"]
    return [value]


def _seats(value: str | None) -> list[str]:
    if value in {None, "driver"}:
        return ["driver"]
    if value == "all":
        return ["driver", "passenger", "rear"]
    if value in {"rear", "rear_left", "rear_right"}:
        return ["rear"]
    return [value]


def _windows(value: str | None) -> list[str]:
    all_windows = ["driver", "passenger", "rear_left", "rear_right"]
    return all_windows if value in {None, "all"} else [value]


def _bounded(value: int | float, low: int | float, high: int | float):
    return max(low, min(high, value))


# The model-facing tool schema intentionally leaves these fields optional so
# the SLM can represent relative and absolute commands. At the execution
# boundary, however, at least one actionable field is required. A selector
# such as ``zone=driver`` is not itself an executable request.
ACTIONABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "set_temperature": ("value", "delta"),
    "set_fan_speed": ("level", "delta"),
    "control_window": ("action", "position"),
    "set_hud": ("state", "brightness", "height"),
    "set_ambient_light": ("state", "color", "brightness"),
    "set_volume": ("value", "delta", "mute"),
    "set_seat_massage": ("state", "mode", "intensity"),
    "set_screen_brightness": ("value", "delta", "mode"),
    "connect_device": ("device_name",),
}


CONDITIONAL_REQUIRED: dict[tuple[str, str], tuple[str, ...]] = {
    ("control_radio", "tune"): ("frequency_or_station_name",),
    ("control_radio", "save_preset"): ("preset",),
    ("control_radio", "play_preset"): ("preset",),
    ("control_podcast", "set_speed"): ("speed",),
}


class SimulatedHardware:
    """Validate calls against the served tool schema and update a display twin."""

    def __init__(self, tools: list[dict[str, Any]]):
        self._schemas = {
            item["function"]["name"]: item["function"].get("parameters", {})
            for item in tools
            if item.get("function", {}).get("name") != "non_tool"
        }
        self._lock = threading.RLock()
        self._state = _initial_state()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._state["busy"])

    def set_busy(self, busy: bool) -> dict[str, Any]:
        with self._lock:
            self._state["busy"] = bool(busy)
            self._state["mode"] = "busy" if busy else "ready"
            self._append_event(
                self._state,
                "Phần cứng đang bận" if busy else "Phần cứng đã sẵn sàng",
            )
            return self.snapshot()

    def sync_vehicle_status(
        self,
        *,
        running: bool | None = None,
        speed_kph: int | float | None = None,
    ) -> dict[str, Any]:
        """Synchronize simulator telemetry as the real hardware adapter will."""

        with self._lock:
            vehicle = self._state["vehicle"]
            if running is not None:
                vehicle["running"] = bool(running)
            if speed_kph is not None:
                speed = float(speed_kph)
                if not math.isfinite(speed) or speed < 0:
                    raise ValueError("speed_kph must be a finite non-negative number")
                vehicle["speed_kph"] = speed
            self._append_event(
                self._state,
                f"Đồng bộ xe: {'đang chạy' if vehicle['running'] else 'đã dừng'}, "
                f"{vehicle['speed_kph']:g} km/h",
            )
            return self.snapshot()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._state = _initial_state()
            return self.snapshot()

    def execute(self, call: dict[str, Any]) -> ExecutionResult:
        if not isinstance(call, dict):
            with self._lock:
                return self._result(
                    "",
                    ExecutionStatus.REJECTED,
                    {},
                    error="invalid_tool_call",
                    details={"expected": "object", "received": type(call).__name__},
                )
        raw_name = call.get("name")
        name = raw_name if isinstance(raw_name, str) else ""
        raw_arguments = call.get("arguments", {})
        if raw_arguments is None:
            raw_arguments = {}
        if not isinstance(raw_arguments, dict):
            with self._lock:
                return self._result(
                    name,
                    ExecutionStatus.REJECTED,
                    {},
                    error="invalid_arguments_type",
                    details={
                        "expected": "object",
                        "received": type(raw_arguments).__name__,
                    },
                )
        arguments = copy.deepcopy(raw_arguments)
        with self._lock:
            schema = self._schemas.get(name)
            if schema is None:
                return self._result(name, ExecutionStatus.REJECTED, arguments, error="unknown_tool")

            allowed = set(schema.get("properties", {}))
            unexpected = tuple(sorted(set(arguments) - allowed))
            if unexpected:
                return self._result(
                    name,
                    ExecutionStatus.REJECTED,
                    arguments,
                    error="unknown_arguments",
                    details={"fields": list(unexpected), "allowed": sorted(allowed)},
                )

            missing = tuple(
                field
                for field in schema.get("required", [])
                if field not in arguments or arguments[field] is None
            )
            if missing:
                return self._result(
                    name,
                    ExecutionStatus.MISSING_REQUIRED,
                    arguments,
                    missing=missing,
                    error="missing_required",
                )

            actionable = ACTIONABLE_FIELDS.get(name)
            if actionable and not any(arguments.get(field) is not None for field in actionable):
                semantic_field = (
                    actionable[0]
                    if len(actionable) == 1
                    else "_or_".join(actionable)
                )
                return self._result(
                    name,
                    ExecutionStatus.MISSING_REQUIRED,
                    arguments,
                    missing=(semantic_field,),
                    error="missing_actionable_field",
                    details={"one_of": list(actionable)},
                )

            conditional = CONDITIONAL_REQUIRED.get((name, str(arguments.get("action"))))
            if conditional:
                conditional_missing = self._conditional_missing(arguments, conditional)
                if conditional_missing:
                    return self._result(
                        name,
                        ExecutionStatus.MISSING_REQUIRED,
                        arguments,
                        missing=conditional_missing,
                        error="missing_conditional_field",
                    )

            validation_arguments = self._validation_arguments(name, arguments)
            errors = sorted(
                jsonschema.Draft202012Validator(schema).iter_errors(validation_arguments),
                key=lambda error: list(error.path),
            )
            if errors:
                required = []
                for error in errors:
                    if error.validator == "required":
                        required.extend(
                            field
                            for field in error.validator_value
                            if field not in error.instance
                        )
                if required:
                    return self._result(
                        name,
                        ExecutionStatus.MISSING_REQUIRED,
                        arguments,
                        missing=tuple(dict.fromkeys(required)),
                        error="missing_required",
                    )
                return self._result(
                    name,
                    ExecutionStatus.REJECTED,
                    arguments,
                    error="invalid_arguments",
                    details=self._validation_details(errors[0]),
                )

            if self._state["busy"]:
                return self._result(name, ExecutionStatus.BUSY, arguments, error="hardware_busy")

            condition = self._check_condition(name, arguments)
            if condition is not None:
                code, details = condition
                return self._result(
                    name,
                    ExecutionStatus.REJECTED,
                    arguments,
                    error=code,
                    details=details,
                )

            # Execute against a copy and publish it only after the whole action
            # succeeds. Rejections and internal errors can never partially
            # mutate the digital twin.
            candidate = copy.deepcopy(self._state)
            try:
                self._apply(candidate, name, arguments)
            except Exception as exc:
                return self._result(
                    name,
                    ExecutionStatus.REJECTED,
                    arguments,
                    error="execution_error",
                    details={"type": type(exc).__name__, "message": str(exc)},
                )
            candidate["action_count"] += 1
            candidate["last_action"] = {
                "name": name,
                "arguments": copy.deepcopy(arguments),
            }
            self._append_event(candidate, self._summarize(name, arguments))
            self._state = candidate
            return self._result(name, ExecutionStatus.SUCCESS, arguments)

    @staticmethod
    def _conditional_missing(
        arguments: dict[str, Any],
        fields: tuple[str, ...],
    ) -> tuple[str, ...]:
        missing = []
        for field in fields:
            alternatives = field.split("_or_")
            if not any(arguments.get(alternative) is not None for alternative in alternatives):
                missing.append(field)
        return tuple(missing)

    @staticmethod
    def _validation_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        validation = copy.deepcopy(arguments)
        if name == "set_temperature" and validation.get("unit") == "fahrenheit":
            if isinstance(validation.get("value"), (int, float)) and not isinstance(
                validation.get("value"), bool
            ):
                validation["value"] = round((validation["value"] - 32) * 5 / 9)
            if isinstance(validation.get("delta"), (int, float)) and not isinstance(
                validation.get("delta"), bool
            ):
                validation["delta"] = round(validation["delta"] * 5 / 9)
        return validation

    @staticmethod
    def _validation_details(error: jsonschema.ValidationError) -> dict[str, Any]:
        details: dict[str, Any] = {
            "message": error.message,
            "validator": error.validator,
            "field": str(next(iter(error.path), "")),
        }
        if error.validator in {"minimum", "maximum", "enum", "type"}:
            details[error.validator] = error.validator_value
        return details

    def _check_condition(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, dict[str, Any]] | None:
        vehicle = self._state["vehicle"]
        if (
            name == "control_trunk"
            and arguments.get("action") == "open"
            and (vehicle["running"] or vehicle["speed_kph"] > 0)
        ):
            return (
                "vehicle_not_stopped",
                {
                    "running": vehicle["running"],
                    "speed_kph": vehicle["speed_kph"],
                    "required": "vehicle_stopped",
                },
            )
        return None

    def _result(
        self,
        name: str,
        status: ExecutionStatus,
        arguments: dict[str, Any],
        *,
        missing: tuple[str, ...] = (),
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            name=name,
            status=status,
            arguments=copy.deepcopy(arguments),
            missing=missing,
            error=error,
            details=copy.deepcopy(details) if details else None,
            state=self.snapshot(),
        )

    @staticmethod
    def _append_event(state: dict[str, Any], text: str) -> None:
        state["events"].append(text)
        del state["events"][:-12]

    @staticmethod
    def _summarize(name: str, arguments: dict[str, Any]) -> str:
        values = ", ".join(f"{key}={value}" for key, value in arguments.items())
        return f"{name}({values})" if values else f"{name}()"

    def _apply(self, state: dict[str, Any], name: str, args: dict[str, Any]) -> None:  # noqa: C901
        climate = state["climate"]
        if name == "set_temperature":
            for zone in _zones(args.get("zone")):
                current = climate["temperature"][zone]
                value = args.get("value", current + args.get("delta", 0))
                if args.get("unit") == "fahrenheit":
                    value = (
                        (value - 32) * 5 / 9
                        if "value" in args
                        else current + args.get("delta", 0) * 5 / 9
                    )
                climate["temperature"][zone] = int(_bounded(value, 16, 30))
            climate["power"] = "on"
        elif name == "set_fan_speed":
            climate["fan"] = int(_bounded(args.get("level", climate["fan"] + args.get("delta", 0)), 0, 7))
        elif name == "set_fan_direction":
            direction = args.get("direction") or []
            climate["direction"] = direction if isinstance(direction, list) else [direction]
        elif name == "set_air_recirculation":
            climate["recirculation"] = args.get("mode", climate["recirculation"])
        elif name == "set_climate_power":
            climate["power"] = args.get("state", climate["power"])
            climate["ac"] = bool(args.get("ac", climate["power"] == "on"))
        elif name == "set_ambient_light":
            ambient = state["lighting"]["ambient"]
            for field in ("state", "color", "brightness"):
                if field in args:
                    ambient[field] = args[field]
            if "color" in args or "brightness" in args:
                ambient["state"] = "on"
        elif name == "set_cabin_light":
            cabin = state["lighting"]["cabin"]
            cabin.update({key: args[key] for key in ("state", "brightness") if key in args})
        elif name == "set_fog_lights":
            state["lighting"]["fog"] = "off" if args.get("state") == "off" else args.get("position", "both")
        elif name == "set_auto_high_beam":
            state["lighting"]["auto_high_beam"] = args.get("state", "off")
        elif name == "set_volume":
            media = state["media"]
            media["volume"] = int(_bounded(args.get("value", media["volume"] + args.get("delta", 0)), 0, 100))
            if "mute" in args:
                media["muted"] = bool(args["mute"])
        elif name == "select_media_source":
            state["media"]["source"] = args.get("source", state["media"]["source"])
        elif name == "play_media":
            state["media"].update({"title": args.get("title"), "playback": "playing"})
            if args.get("source") not in {None, "any"}:
                state["media"]["source"] = args["source"]
        elif name == "control_playback":
            state["media"]["playback"] = {
                "play": "playing", "pause": "paused", "stop": "stopped",
                "restart": "playing", "next": "playing", "previous": "playing",
            }.get(args.get("action"), state["media"]["playback"])
        elif name in {"set_seat_heating", "set_seat_ventilation"}:
            key = "seat_heating" if name.endswith("heating") else "seat_ventilation"
            for seat in _seats(args.get("seat")):
                state["comfort"][key][seat] = int(_bounded(args.get("level", 0), 0, 3))
        elif name == "set_seat_massage":
            value = "off" if args.get("state") == "off" else args.get("mode", "on")
            for seat in _seats(args.get("seat")):
                state["comfort"]["massage"][seat] = value
        elif name == "control_window":
            value = args.get("position")
            if value is None:
                value = 100 if args.get("action") == "open" else 0
            for window in _windows(args.get("window")):
                state["access"]["windows"][window] = int(_bounded(value, 0, 100))
        elif name == "control_sunroof":
            value = args.get("position")
            if value is None:
                value = {"open": 100, "close": 0, "vent": 15, "tilt": 10}.get(args.get("action"), 0)
            state["access"]["sunroof"] = int(_bounded(value, 0, 100))
        elif name == "control_trunk":
            state["access"]["trunk"] = "open" if args.get("action") == "open" else "closed"
        elif name == "control_mirrors":
            state["access"]["mirrors"] = "folded" if args.get("action") == "fold" else "unfolded"
        elif name == "set_drive_mode":
            state["drive"]["mode"] = args.get("mode", state["drive"]["mode"])
        elif name == "set_adas_setting":
            feature = args.get("feature")
            state["drive"]["adas"][feature] = args.get("state", args.get("level", "on"))
        elif name == "set_regen_braking":
            state["drive"]["regen"] = args.get("level", state["drive"]["regen"])
        elif name == "set_connectivity":
            state["connectivity"][args.get("interface")] = args.get("state")
        elif name == "connect_device":
            state["connectivity"]["device"] = args.get("device_name")
        else:
            state["other"][name] = copy.deepcopy(args)
