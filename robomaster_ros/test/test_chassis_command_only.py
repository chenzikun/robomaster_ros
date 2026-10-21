from robomaster_ros.modules.chassis import (
    CHASSIS_WORK_MODE_IDLE,
    CHASSIS_WORK_MODE_SDK,
    Chassis,
)


class FakeChassisApi:
    def __init__(self) -> None:
        self.work_modes = []
        self.wheel_commands = []
        self.speed_commands = []

    def _send_sync_proto(self, proto) -> bool:
        self.work_modes.append(proto._mode)
        return True

    def drive_wheels(self, *speeds, timeout=None) -> bool:
        self.wheel_commands.append((speeds, timeout))
        return True

    def drive_speed(self, *, x, y, z, timeout=None) -> bool:
        self.speed_commands.append(((x, y, z), timeout))
        return True


def make_chassis() -> Chassis:
    chassis = object.__new__(Chassis)
    chassis.api = FakeChassisApi()
    chassis.command_only = True
    chassis.timeout = 0.5
    chassis._chassis_work_mode = None
    chassis._chassis_command_active = False
    return chassis


def test_command_only_starts_in_idle_mode() -> None:
    chassis = make_chassis()

    chassis._set_chassis_work_mode(CHASSIS_WORK_MODE_IDLE)

    assert chassis.api.work_modes == [CHASSIS_WORK_MODE_IDLE]
    assert chassis.api.wheel_commands == []
    assert chassis.api.speed_commands == []


def test_command_only_returns_to_idle_after_motion() -> None:
    chassis = make_chassis()

    chassis._ensure_chassis_work_mode()
    chassis._chassis_command_active = True
    chassis._enter_chassis_idle_mode()

    assert chassis.api.work_modes == [CHASSIS_WORK_MODE_SDK, CHASSIS_WORK_MODE_IDLE]
    assert chassis.api.wheel_commands == [((0, 0, 0, 0), 0.5)]
    assert chassis.api.speed_commands == [((0.0, 0.0, 0.0), 0.5)]
    assert chassis._chassis_command_active is False
