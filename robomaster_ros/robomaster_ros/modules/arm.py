import threading
import time
import math

import rclpy.action
import robomaster.robot
import robomaster.robotic_arm
import robomaster.action

import geometry_msgs.msg
import robomaster_msgs.action
import sensor_msgs.msg
import rcl_interfaces.msg

from .servo import ServoData, RMServo

from typing import Dict, Tuple, Any, Optional, cast
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..client import RoboMasterROS

from .. import Module
from ..action import wait_action, abort_action, remove_action_with_target
from ..utils import rate, nearest_rate, Rate

SERVO_DIRECT_DEFAULT_PULSE_SEC = 0.18
SERVO_DIRECT_DEFAULT_GAIN = 2200.0
SERVO_DIRECT_DEFAULT_MAX_SPEED = 20.0
SERVO_DIRECT_ZERO_EPS_M = 1e-6
SERVO_DIRECT_MIN_PULSE_SEC = 0.05
ARM_MM_PER_M = 1000.0


def arm_joint_state(left_motor: float, right_motor: float) -> Dict[str, float]:
    # Now that the urdf defines dependent joints, we can skip most of them
    return {
        'arm_1_joint': right_motor,
        'rod_joint': left_motor,
        'rod_3_joint': right_motor - left_motor,
    }


class Arm(Module):

    def __init__(self, robot: robomaster.robot.Robot, node: 'RoboMasterROS') -> None:
        self.node = node
        self.robot = robot
        right_motor_zero: int = node.declare_parameter('arm.right_motor.zero', 1024).value
        left_motor_zero: int = node.declare_parameter('arm.left_motor.zero', 1024).value
        right_motor_angle: float = node.declare_parameter('arm.right_motor.angle', -0.274016).value
        left_motor_angle: float = node.declare_parameter('arm.left_motor.angle', 0.073304).value
        right_motor_direction: int = node.declare_parameter('arm.right_motor.direction', -1).value
        left_motor_direction: int = node.declare_parameter('arm.left_motor.direction', 1).value
        right_motor_index: int = node.declare_parameter('arm.right_motor.index', 1).value
        left_motor_index: int = node.declare_parameter('arm.left_motor.index', 0).value
        self.servo_direct_enabled: bool = node.declare_parameter(
            'arm.servo_direct.enabled', True).value
        self.servo_direct_pulse_sec: float = node.declare_parameter(
            'arm.servo_direct.pulse_sec', SERVO_DIRECT_DEFAULT_PULSE_SEC).value
        self.servo_direct_gain_x: float = node.declare_parameter(
            'arm.servo_direct.gain_x', SERVO_DIRECT_DEFAULT_GAIN).value
        self.servo_direct_gain_z: float = node.declare_parameter(
            'arm.servo_direct.gain_z', SERVO_DIRECT_DEFAULT_GAIN).value
        self.servo_direct_max_speed: float = node.declare_parameter(
            'arm.servo_direct.max_speed', SERVO_DIRECT_DEFAULT_MAX_SPEED).value
        self.servo_direct_left_x_sign: int = node.declare_parameter(
            'arm.servo_direct.left_x_sign', 1).value
        self.servo_direct_right_x_sign: int = node.declare_parameter(
            'arm.servo_direct.right_x_sign', -1).value
        self.servo_direct_left_z_sign: int = node.declare_parameter(
            'arm.servo_direct.left_z_sign', 1).value
        self.servo_direct_right_z_sign: int = node.declare_parameter(
            'arm.servo_direct.right_z_sign', 1).value
        self._servo_direct_stop_timer: Optional[threading.Timer] = None
        self.servos = {
            'right_motor': RMServo(
                index=right_motor_index, reference_angle=right_motor_angle,
                reference_value=right_motor_zero, direction=right_motor_direction,
                name=node.tf_frame('arm_1_joint')),
            'left_motor': RMServo(
                index=left_motor_index, reference_angle=left_motor_angle,
                reference_value=left_motor_zero, direction=left_motor_direction,
                name=node.tf_frame("rod_joint"))
        }
        node.get_logger().info(
            f"[Arm] left servo: index {left_motor_index}, direction {left_motor_direction}, "
            f"value {left_motor_zero}, angle {left_motor_angle}")
        node.get_logger().info(
            f"[Arm] right servo: index {right_motor_index}, direction {right_motor_direction}, "
            f"value {right_motor_zero}, angle {right_motor_angle}")
        node.get_logger().info(
            f"[Arm] servo_direct={self.servo_direct_enabled} gain_x={self.servo_direct_gain_x} "
            f"gain_z={self.servo_direct_gain_z} max_speed={self.servo_direct_max_speed}")
        self.arm_position_msg = geometry_msgs.msg.PointStamped()
        self.arm_position_msg.header.frame_id = node.tf_frame('arm_base_link')
        self.arm_position_msg.point.y = 0.0
        self.arm_position_pub = node.create_publisher(
            geometry_msgs.msg.PointStamped, 'arm_position', 1)
        self.robot = robot
        self.clock = node.get_clock()
        self.logger = node.get_logger()
        self.node = node
        self.api = robot.robotic_arm
        arm_rate = rate(node, 'arm', 10)
        if arm_rate:
            self.subscribe(arm_rate)
            self.servo_raw_state_pub = node.create_publisher(
                robomaster_msgs.msg.ServoRawState, 'servo_raw_state', 1)
        cbg = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self._move_arm_action_server = rclpy.action.ActionServer(
            node, robomaster_msgs.action.MoveArm, 'move_arm', self.execute_move_arm_callback,
            goal_callback=self.new_move_arm_goal_callback,
            cancel_callback=self.cancel_move_arm_callback, callback_group=cbg)
        self.action: Optional[robomaster.action.Action] = None
        node.create_subscription(geometry_msgs.msg.Vector3, 'cmd_arm',
                                 self.has_received_velocity_command, 10)
        node.create_subscription(geometry_msgs.msg.Point, 'target_arm_position',
                                 self.has_received_position_command, 1)
        node.add_on_set_parameters_callback(self.set_params_cb)

    def _start_arm_action(self, x_mm: int, z_mm: int, *, relative: bool) -> None:
        """Start arm motion through the SDK public API; raw client messages are ignored on EP."""
        if self.action:
            try:
                abort_action(self.robot, self.action)
            except Exception as exc:
                self.logger.warning(f'[Arm] Failed aborting previous action: {exc}')
        try:
            self.robot.set_robot_mode(robomaster.robot.FREE)
        except Exception as exc:
            self.logger.warning(f'[Arm] Failed setting robot mode free before move: {exc}')
        action_fn = self.api.move if relative else self.api.moveto
        try:
            self.action = action_fn(x=x_mm, y=z_mm)
            self.logger.info(
                f"[Arm] Start {'relative' if relative else 'absolute'} move x={x_mm} y={z_mm}: "
                f"{self.action}"
            )
        except RuntimeError as exc:
            self.logger.warning(f'[Arm] Failed creating arm action {exc}; remove current action')
            remove_action_with_target(
                self.robot, robomaster.robotic_arm.RoboticArmMoveAction._target)
            self.action = action_fn(x=x_mm, y=z_mm)
        timeout = self._arm_action_timeout(x_mm, z_mm, relative=relative)
        threading.Thread(
            target=self._wait_current_arm_action,
            args=(self.action, timeout),
            daemon=True,
        ).start()

    def _arm_action_timeout(self, x_mm: int, z_mm: int, *, relative: bool) -> float:
        if relative:
            distance_m = max(abs(x_mm), abs(z_mm)) * 0.001
        else:
            cur_x = getattr(self, 'x', x_mm)
            cur_z = getattr(self, 'z', z_mm)
            distance_m = max(abs(x_mm - cur_x), abs(z_mm - cur_z)) * 0.001
        return max(1.0, distance_m / 0.02)

    def _wait_current_arm_action(self, action: robomaster.action.Action, timeout: float) -> None:
        def cb() -> None:
            return None

        ok = wait_action(action, cb, timeout=timeout)
        self.logger.info(f'[Arm] Action finished ok={ok}: {action}')
        if self.action is action:
            self.action = None

    def has_received_velocity_command(self, msg: geometry_msgs.msg.Vector3) -> None:
        if self.servo_direct_enabled:
            self._drive_arm_servos(float(msg.x), float(msg.z))
            return
        x = int(msg.x * 1000)
        z = int(msg.z * 1000)
        if x == 0 and z == 0:
            return
        self._start_arm_action(x, z, relative=True)

    def has_received_position_command(self, msg: geometry_msgs.msg.Point) -> None:
        if self.servo_direct_enabled:
            cur_x = getattr(self, 'x', None)
            cur_z = getattr(self, 'z', None)
            if cur_x is None or cur_z is None:
                self.logger.warning('[Arm] servo direct target ignored: no arm position feedback yet')
                return
            dx = float(msg.x) - float(cur_x) * 0.001
            dz = float(msg.z) - float(cur_z) * 0.001
            self._drive_arm_servos(dx, dz)
            return
        x = int(msg.x * 1000)
        z = int(msg.z * 1000)
        self._start_arm_action(x, z, relative=False)

    def _drive_arm_servos(self, dx_m: float, dz_m: float) -> None:
        if abs(dx_m) < SERVO_DIRECT_ZERO_EPS_M and abs(dz_m) < SERVO_DIRECT_ZERO_EPS_M:
            self._pause_arm_servos()
            return
        left_speed = self._clamp_servo_speed(
            self.servo_direct_left_x_sign * dx_m * self.servo_direct_gain_x
            + self.servo_direct_left_z_sign * dz_m * self.servo_direct_gain_z)
        right_speed = self._clamp_servo_speed(
            self.servo_direct_right_x_sign * dx_m * self.servo_direct_gain_x
            + self.servo_direct_right_z_sign * dz_m * self.servo_direct_gain_z)
        self.robot.servo.drive_speed(
            index=self.servos['left_motor'].index + 1, speed=left_speed)
        self.robot.servo.drive_speed(
            index=self.servos['right_motor'].index + 1, speed=right_speed)
        self._schedule_servo_pause()

    def _clamp_servo_speed(self, speed: float) -> float:
        if not math.isfinite(speed):
            return 0.0
        limit = abs(float(self.servo_direct_max_speed))
        return max(-limit, min(limit, float(speed)))

    def _schedule_servo_pause(self) -> None:
        if self._servo_direct_stop_timer:
            self._servo_direct_stop_timer.cancel()
        self._servo_direct_stop_timer = threading.Timer(
            max(SERVO_DIRECT_MIN_PULSE_SEC, float(self.servo_direct_pulse_sec)), self._pause_arm_servos)
        self._servo_direct_stop_timer.daemon = True
        self._servo_direct_stop_timer.start()

    def _pause_arm_servos(self) -> None:
        if self._servo_direct_stop_timer:
            self._servo_direct_stop_timer.cancel()
            self._servo_direct_stop_timer = None
        self.robot.servo.pause(index=self.servos['left_motor'].index + 1)
        self.robot.servo.pause(index=self.servos['right_motor'].index + 1)

    def subscribe(self, rate: Rate) -> None:
        if rate:
            self.api.sub_position(freq=rate, callback=self.updated_arm_position)
            self.robot.servo.sub_servo_info(freq=rate, callback=self.updated_arm_servo)

    def set_params_cb(self, params: Any) -> rcl_interfaces.msg.SetParametersResult:
        for param in params:
            if 'arm' not in param.name:
                continue
            if param.name == 'arm.rate':
                param.value = nearest_rate(param.value)
                self.subscribe(param.value)
            for name, servo in self.servos.items():
                if name in param.name:
                    if param.name == f'arm.{name}.zero':
                        servo.reference_value = param.value
                    elif param.name == f'arm.{name}.angle':
                        servo.reference_angle = param.value
                    elif param.name == f'arm.{name}.direction':
                        servo.direction = 1 if param.value > 0 else -1
        return rcl_interfaces.msg.SetParametersResult(successful=True)

    def stop(self) -> None:
        self._pause_arm_servos()
        self._move_arm_action_server.destroy()
        if self.node.connected:
            self.api.unsub_position()
            self.robot.servo.unsub_servo_info()

    def abort(self) -> None:
        if self.action:
            self.action._abort()
            while self.action is not None:
                self.logger.info(f"[Arm] wait for action {self.action} to terminate")
                time.sleep(0.1)

    def publish_sensor_raw_state(self, msg: ServoData) -> None:
        omsg = robomaster_msgs.msg.ServoRawState()
        for i in range(4):
            omsg.valid[i] = (msg[0][i] != 0)
            omsg.speed[i] = msg[1][i]
            omsg.value[i] = msg[2][i]
        self.servo_raw_state_pub.publish(omsg)

    # (valid, speed, angle)
    def updated_arm_servo(self, msg: ServoData) -> None:
        for v in self.servos.values():
            v.valid = (msg[0][v.index] != 0)
            if v.valid:
                v.value = msg[2][v.index]
        right_motor = self.servos['right_motor']
        left_motor = self.servos['left_motor']
        if (right_motor.valid and left_motor.valid):
            arm_state_msg = sensor_msgs.msg.JointState()
            arm_state_msg.header.stamp = self.clock.now().to_msg()
            joints = arm_joint_state(right_motor=right_motor.angle, left_motor=left_motor.angle)
            arm_state_msg.name = [self.node.tf_frame(name) for name in joints.keys()]
            arm_state_msg.position = list(joints.values())
            self.node.joint_state_pub.publish(arm_state_msg)
        for servo in self.servos.values():
            servo.valid = False
        self.publish_sensor_raw_state(msg)

    # TODO: actions are NOT threaded -> they block other callbacks
    # so the hearbeat deadman trigger and the node exit
    # Actions should not block! ... I should keep receving and forwarding data
    def execute_move_arm_callback(self, goal_handle: Any) -> robomaster_msgs.action.MoveArm.Result:
        # TODO(jerome): Complete with failures, ...
        request = goal_handle.request
        timeout: float = 5.0
        if request.relative:
            f = self.api.move
            delta = max(abs(request.x), abs(request.z))
            if delta == 0:
                # HACK(Jerome): else it moves to an undefined target
                goal_handle.succeed()
                self.logger.info('[Arm] No need to move')
                return robomaster_msgs.action.MoveArm.Result()
            # HACK(Jerome): else the teleop +/- x is not concluding the action
            timeout = max(1.0, delta / 0.02)  # 2cm/s
        else:
            f = self.api.moveto
        while True:
            try:
                new_action = f(x=int(request.x * 1000), y=int(request.z * 1000))
                break
            except RuntimeError as e:
                self.logger.warning(
                    f'[Arm] Failed creating new action {e} => remove current action')
                remove_action_with_target(
                    self.robot, robomaster.robotic_arm.RoboticArmMoveAction._target)
        self.action = new_action
        self.logger.info(f'[Arm] Start moving with request {request}: {new_action}')
        feedback_msg = robomaster_msgs.action.MoveArm.Feedback()

        def cb() -> None:
            feedback_msg.progress = cast(robomaster.action.Action, new_action)._percent * 0.01
            goal_handle.publish_feedback(feedback_msg)

        wait_action(new_action, cb, timeout=timeout)

        if goal_handle.is_cancel_requested:
            self.logger.info('[Arm] Done moving: canceled')
            goal_handle.canceled()
        elif new_action.has_succeeded:
            self.logger.info('[Arm] Done moving: succeeded')
            goal_handle.succeed()
        else:
            self.logger.warning('[Arm] Done moving: aborted')
            try:
                goal_handle.abort()
            except rclpy._rclpy_pybind11.RCLError:
                # Happens when the context shutdown before we send feedbacks
                pass
        self.action = None
        return robomaster_msgs.action.MoveArm.Result()

    def new_move_arm_goal_callback(self, goal_request: robomaster_msgs.action.MoveArm.Goal
                                   ) -> rclpy.action.server.GoalResponse:
        if self.action:
            return rclpy.action.server.GoalResponse.REJECT
        return rclpy.action.server.GoalResponse.ACCEPT

    def cancel_move_arm_callback(self, goal_handle: Any) -> rclpy.action.CancelResponse:
        if self.action:
            self.node.executor.create_task(self._stop_action)
        return rclpy.action.CancelResponse.ACCEPT

    def _stop_action(self) -> None:
        if self.action:
            self.logger.info('[Arm] Canceling action')
            abort_action(self.robot, self.action)

    # (x, z) in mm
    def updated_arm_position(self, msg: Tuple[int, int]) -> None:
        self.arm_position_msg.header.stamp = self.clock.now().to_msg()
        ps = self.arm_position_msg.point
        ps.x, ps.z = [v * 0.001 for v in msg]
        self.x, self.z = msg
        self.arm_position_pub.publish(self.arm_position_msg)
