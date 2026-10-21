from robomaster_ros.modules.arm import Arm


class FakeServo:
    def __init__(self) -> None:
        self.calls = []

    def drive_speed(self, *, index: int, speed: float) -> bool:
        self.calls.append((index, speed))
        return True


class FakeRobot:
    def __init__(self) -> None:
        self.servo = FakeServo()


def make_arm() -> Arm:
    arm = object.__new__(Arm)
    arm.robot = FakeRobot()
    arm.servos = {
        'left_motor': type('ServoState', (), {'index': 0})(),
        'right_motor': type('ServoState', (), {'index': 1})(),
    }
    arm.servo_direct_gain_x = 100.0
    arm.servo_direct_gain_z = 100.0
    arm.servo_direct_left_x_sign = 1
    arm.servo_direct_right_x_sign = -1
    arm.servo_direct_left_z_sign = 1
    arm.servo_direct_right_z_sign = 1
    arm.servo_direct_max_speed = 20.0
    arm._schedule_servo_pause = lambda: None
    return arm


def test_servo_direct_drives_both_servos_with_signed_speed() -> None:
    arm = make_arm()

    arm._drive_arm_servos(0.1, 0.0)

    assert arm.robot.servo.calls == [(1, 10.0), (2, -10.0)]


def test_servo_direct_clamps_combined_axis_speed() -> None:
    arm = make_arm()

    arm._drive_arm_servos(1.0, 1.0)

    assert arm.robot.servo.calls == [(1, 20.0), (2, 0.0)]
