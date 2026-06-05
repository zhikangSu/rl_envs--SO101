import logging
import time
from collections import deque
import numpy as np
from gymnasium import Env, spaces
import gymnasium as gym
from scipy.spatial.transform import Rotation
from gymnasium.spaces import Box
from gymnasium.spaces import flatten_space, flatten
from rl_envs.shared_state import shared_state
import cv2
import traceback
import sys

class HumanIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.robot_type = env.unwrapped.robot_type
        self.dual_arm = env.unwrapped.dual_arm
        self.env.unwrapped.init_xtele()  # init xtele
        self.control_mode = env.unwrapped.control_mode # 控制模式
    

    def reset(self, **kwargs):
        """Reset the environment and sync robot position."""
        obs, info = self.env.reset(**kwargs)
        shared_state.human_intervention_key = False
        self.env.unwrapped.sync_xtele(timeout=2)
        info["is_intervention"] = False
        return obs, info

    def pose2matrix(self, pose):
        pose_t, pose_quat = pose[0:3], pose[3:7]
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = Rotation.from_quat(pose_quat).as_matrix()
        pose_matrix[:3, 3] = pose_t
        return pose_matrix

    def compute_expert_action(self, curr_pose, target_pose, target_joint):
        curr_matrix = self.pose2matrix(curr_pose)
        tar_matrix = self.pose2matrix(target_pose)
        T_diff_matrix = np.dot(np.linalg.inv(curr_matrix), tar_matrix)
        rel_rot = Rotation.from_matrix(T_diff_matrix[:3, :3]).as_euler("xyz")  # 相对旋转（欧拉角）
        rel_pos = T_diff_matrix[:3, 3]  # 相对位置
        expert_a = np.zeros(7, dtype=np.float64) # xyz+rpy+gripper
        expert_a[:3] = rel_pos / self.env.unwrapped.action_scale[0] # 位置增量
        expert_a[3:6] = rel_rot / self.env.unwrapped.action_scale[1] # 旋转增量
        expert_a[6:] = target_joint[-1] / self.env.unwrapped.action_scale[2] # 夹爪
        
        """
        intervention action 边缘裁剪
        """
        epsilon = 1e-6
        expert_a = np.clip(expert_a, [-1.0+epsilon, -1.0+epsilon, -1.0+epsilon, -1.0+epsilon, -1.0+epsilon, -1.0+epsilon, 0.0], [1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0])
        return expert_a
    

    def action(self, action: np.ndarray) -> np.ndarray:
        intervened = shared_state.human_intervention_key
        if intervened:
            try:
                obs = self.env.unwrapped.get_xtele()
                xtele_joints, xtele_pose = obs['joints'], obs['pose']

                if self.control_mode == "joint":
                    expert_a = xtele_joints
                else:
                    if self.dual_arm:
                        if "tienkung" in self.robot_type:
                            expert_a = []
                            # 逐臂计算
                            for name, single_target_pose in xtele_pose.items():
                                single_curr_pose = self.env.unwrapped.currpos[name]
                                single_expert_a = self.compute_expert_action(single_curr_pose, single_target_pose, xtele_joints[name])
                                expert_a += single_expert_a.tolist()
                            expert_a = np.array(expert_a)
                        else:
                            raise NotImplementedError("Unknown robot type")
                    else:
                        curr_pose = self.env.unwrapped.currpos
                        expert_a = self.compute_expert_action(curr_pose, xtele_pose, xtele_joints)

                return expert_a, xtele_joints, True
            except Exception as e:
                print(f"Error in action: {e}")
                print(f"[{type(e).__name__}] {e!r}")
                traceback.print_exc()          # full stacktrace
                sys.exit(1)
        return action, None, False

    def step(self, action):
        action, xtele_joints,replaced = self.action(action)
        if replaced:
            obs, rew, terminated, truncated, info = self.env.step(action)
            info["intervene_action"] = action
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)        
            self.env.unwrapped.sync_xtele(timeout=0.1)
        

        info["is_intervention"] = replaced
        return obs, rew, terminated, truncated, info



class SpaceMouseIntervention(gym.ActionWrapper):
    """Override policy actions with SpaceMouse actions when operator input is detected."""

    def __init__(
        self,
        env,
        action_indices=None,
        deadzone=1e-3,
        axis_deadzone=None,
        enable_gripper=True,
        expert=None,
        translation_scale=1.0,
        rotation_scale=1.0,
        axis_signs=None,
    ):
        super().__init__(env)
        self.action_indices = action_indices
        self.deadzone = float(deadzone)
        self.axis_deadzone = float(axis_deadzone if axis_deadzone is not None else deadzone)
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        if axis_signs is None:
            axis_signs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        axis_signs = np.asarray(axis_signs, dtype=np.float32).reshape(-1)
        if axis_signs.size != 6:
            raise ValueError(f"spacemouse_axis_signs must have 6 values, got {axis_signs.size}")
        self.axis_signs = axis_signs
        self.gripper_enabled = bool(enable_gripper) and int(self.action_space.shape[0]) >= 7
        self.left = False
        self.right = False

        if expert is None:
            try:
                from rl_envs.spacemouse.spacemouse_expert import SpaceMouseExpert
            except Exception as exc:
                raise ImportError(
                    "Failed to import SpaceMouse runtime. Install dependency 'easyhid' "
                    "and verify HID permissions for SpaceMouse access."
                ) from exc

            self.expert = SpaceMouseExpert()
        else:
            self.expert = expert

    def _get_expert_action(self):
        expert_a, buttons = self.expert.get_action()
        expert_a = np.asarray(expert_a, dtype=np.float32).reshape(-1)
        buttons = list(buttons) if buttons is not None else []

        if expert_a.size >= 6:
            motion = expert_a[:6].copy()
        else:
            motion = np.zeros((6,), dtype=np.float32)
            motion[:expert_a.size] = expert_a

        # Calibrate SpaceMouse direction/sensitivity in one place.
        motion[:3] = motion[:3] * self.translation_scale
        motion[3:6] = motion[3:6] * self.rotation_scale
        motion = motion * self.axis_signs
        # Suppress small per-axis jitter so idle hand does not move the robot.
        motion[np.abs(motion) < self.axis_deadzone] = 0.0
        expert_a = motion

        if len(buttons) >= 2:
            self.left, self.right = bool(buttons[0]), bool(buttons[1])
        else:
            self.left, self.right = False, False

        intervened = np.linalg.norm(expert_a) > self.deadzone

        if self.gripper_enabled:
            if self.left:
                gripper_action = np.array([-1.0], dtype=np.float32)
                intervened = True
            elif self.right:
                gripper_action = np.array([1.0], dtype=np.float32)
                intervened = True
            else:
                gripper_action = np.array([0.0], dtype=np.float32)

            if expert_a.shape[0] >= 6:
                expert_a = np.concatenate((expert_a[:6], gripper_action), axis=0)
            else:
                expert_a = np.concatenate((np.zeros((6,), dtype=np.float32), gripper_action), axis=0)

        target_dim = int(self.action_space.shape[0])
        if expert_a.shape[0] < target_dim:
            expert_a = np.pad(expert_a, (0, target_dim - expert_a.shape[0]), mode="constant")
        elif expert_a.shape[0] > target_dim:
            expert_a = expert_a[:target_dim]

        if self.action_indices is not None:
            filtered = np.zeros_like(expert_a)
            filtered[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered

        return expert_a.astype(np.float32), intervened

    def step(self, action):
        expert_action, replaced = self._get_expert_action()
        exec_action = expert_action if replaced else action

        obs, rew, terminated, truncated, info = self.env.step(exec_action)
        info["spacemouse_action_norm"] = float(np.linalg.norm(expert_action[:6]))
        info["executed_action"] = np.asarray(exec_action, dtype=np.float32)
        if replaced:
            info["intervene_action"] = np.asarray(exec_action, dtype=np.float32)
        info["is_intervention"] = bool(replaced)
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, terminated, truncated, info

    def close(self):
        if hasattr(self.expert, "close"):
            self.expert.close()
        return self.env.close()


class SO101LeaderIntervention(gym.ActionWrapper):
    """SO101 leader-arm intervention.

    Default mode is policy-autonomous: the policy command is sent to the follower, and
    the leader is servoed to mirror the follower's current joint state. If the operator
    pulls the leader away from the mirrored pose, the wrapper switches to intervention:
    leader torque is disabled and follower commands are replaced by leader commands.

    In joint mode, takeover sends raw leader joint targets. In SO101 EE-delta mode,
    policy actions stay low-dimensional, but takeover execution uses raw leader
    joint targets so the follower mirrors the whole arm like native teleoperation.
    The transition stored for learning is still converted back to
    [dx, dy, dz, gripper] from the actual follower EE motion.

    Release uses a small state machine instead of a raw threshold. After takeover, the
    wrapper waits until the leader has been still for a few ticks and leader/follower
    error is small, then it mirrors the leader back to the follower and resumes policy.
    """

    def __init__(
        self,
        env,
        error_threshold_deg=8.0,
        gripper_binary_threshold_pct=15.0,
        release_error_threshold_deg=None,
        release_motion_threshold_deg=0.75,
        min_intervention_s=0.5,
        release_queue_size=4,
        ee_error_threshold_m=0.025,
        ee_release_error_threshold_m=0.008,
        manual_takeover_enabled=True,
        auto_takeover_enabled=False,
        release_guard_s=0.6,
        release_settle_timeout_s=1.0,
        hold_p_coefficient=32,
        hold_d_coefficient=32,
    ):
        super().__init__(env)
        self.control_mode = self.env.unwrapped.control_mode
        self.error_threshold = float(error_threshold_deg)
        self.ee_error_threshold = float(ee_error_threshold_m)
        self.ee_release_error_threshold = float(ee_release_error_threshold_m)
        self.manual_takeover_enabled = bool(manual_takeover_enabled)
        self.auto_takeover_enabled = bool(auto_takeover_enabled)
        self.release_guard_s = float(release_guard_s)
        self.release_settle_timeout_s = float(release_settle_timeout_s)
        self.hold_p_coefficient = int(hold_p_coefficient)
        self.hold_d_coefficient = int(hold_d_coefficient)
        self.gripper_binary_threshold = float(gripper_binary_threshold_pct)
        if release_error_threshold_deg is None:
            release_error_threshold_deg = max(2.0, self.error_threshold * 0.5)
        self.release_error_threshold = float(release_error_threshold_deg)
        self.release_motion_threshold = float(release_motion_threshold_deg)
        self.min_intervention_s = float(min_intervention_s)
        self.leader_motion_queue = deque(maxlen=max(1, int(release_queue_size)))
        self.is_intervening = False
        self.intervention_started_at = None
        self.prev_leader_arm = None
        self.leader = None
        self._leader_motor_names = None
        self.leader_torque_enabled = False
        self.manual_intervention = False
        self.is_release_settling = False
        self.release_settle_target = None
        self.release_settle_started_at = None
        self.release_guard_until = 0.0
        self._last_torque_warning_at = 0.0
        self.leader_torque_faulted = False
        self._leader_torque_fault_message = None
        # After a Feetech overload trip, keep the leader limp until this time so the
        # protection latch clears; enable retries automatically afterwards.
        self._torque_retry_after = 0.0
        self._torque_cooldown_s = 3.0
        cfg = self.env.unwrapped.config
        self._station = self.env.unwrapped.robot_station
        self._joint_min = np.asarray(list(cfg.so101_joint_action_min), dtype=np.float32)
        self._joint_max = np.asarray(list(cfg.so101_joint_action_max), dtype=np.float32)
        self._ee_delta_scale = np.asarray(
            list(getattr(cfg, "so101_ee_delta_scale_m", [0.035, 0.030, 0.060])),
            dtype=np.float32,
        )
        # Cache follower so human takeover can temporarily bypass max_relative_target.
        # The cap is restored before policy control resumes; policy EE-delta actions
        # still keep the per-tick joint cap as the final hardware safety net.
        self._follower = self.env.unwrapped.robot_station.follower
        self._saved_max_relative_target = None
        self._setup_leader()

    def _setup_leader(self):
        from lerobot.teleoperators.so101_leader import SO101Leader
        from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig

        cfg = self.env.unwrapped.config
        leader_cfg = SO101LeaderConfig(
            port=str(cfg.so101_leader_port),
            id=str(cfg.so101_leader_id),
        )
        self.leader = SO101Leader(leader_cfg)
        self.leader.connect()
        self._leader_motor_names = list(self.leader.bus.motors)
        for motor in self._leader_motor_names:
            try:
                self.leader.bus.write("P_Coefficient", motor, self.hold_p_coefficient)
                self.leader.bus.write("I_Coefficient", motor, 0)
                self.leader.bus.write("D_Coefficient", motor, self.hold_d_coefficient)
            except Exception as exc:
                logging.warning(
                    "[SO101LeaderIntervention] failed to configure leader gains for %s: %s",
                    motor,
                    exc,
                )

    def _read_leader_joints(self):
        action = self.leader.get_action()
        return np.array(
            [float(action[f"{n}.pos"]) for n in self._leader_motor_names], dtype=np.float32
        )

    def _read_follower_joints(self):
        obs = self.env.unwrapped.robot_station.get_obs()
        return np.asarray(obs["arm_joints"]["single"], dtype=np.float32)

    def _warn_leader_torque(self, message, exc):
        logging.warning("[SO101LeaderIntervention] %s: %s", message, exc)
        now = time.perf_counter()
        if now - self._last_torque_warning_at >= 2.0:
            print(f"[警告] {message}：{exc}", flush=True)
            self._last_torque_warning_at = now

    def _enable_leader_torque(self):
        if self.leader_torque_faulted:
            return False
        if self.leader_torque_enabled:
            return True
        if time.perf_counter() < self._torque_retry_after:
            # Cooling down after a Feetech overload trip: keep the leader limp so the
            # servo's protection latch clears; the next tick retries automatically.
            return False
        try:
            self.leader.bus.enable_torque(num_retry=2)
        except Exception as exc:
            self.leader_torque_enabled = False
            if "overload" in str(exc).lower():
                # Overload protection (usually the elbow under gravity). Relieve the load
                # by forcing torque off and back off for a cooldown, then retry — instead
                # of latching a permanent fault that leaves the arm stuck until restart.
                self._relieve_leader_overload()
                self._torque_retry_after = time.perf_counter() + self._torque_cooldown_s
                self._warn_leader_torque("主臂过载保护触发，已卸力冷却后自动重试", exc)
            else:
                self.leader_torque_faulted = True
                self._leader_torque_fault_message = str(exc)
                self._warn_leader_torque("主臂上扭矩失败，已保持为无扭矩状态", exc)
            return False
        self.leader_torque_enabled = True
        self._torque_retry_after = 0.0
        return True

    def _relieve_leader_overload(self):
        """Force the leader torque off so a Feetech overload protection latch can clear."""
        try:
            self.leader.bus.disable_torque(num_retry=2)
        except Exception:
            pass
        self.leader_torque_enabled = False

    def _disable_leader_torque(self, force=False):
        success = True
        if self.leader_torque_faulted and not self.leader_torque_enabled:
            return False
        if self.leader_torque_enabled or force:
            try:
                self.leader.bus.disable_torque(num_retry=2)
            except Exception as exc:
                success = False
                self.leader_torque_faulted = True
                self._leader_torque_fault_message = str(exc)
                self._warn_leader_torque("主臂关扭矩失败，将继续按无扭矩状态处理", exc)
            finally:
                self.leader_torque_enabled = False
        return success

    def _mirror_leader_to_follower(self):
        """Drive leader to match follower current joint state (active servo).

        Order matters for safety: write Goal_Position BEFORE enabling torque.
        If we enable torque first, the servo will spring to whatever stale Goal_Position
        value is in the firmware register (could be far from current pose), causing a
        sudden jerk before the new goal is written. By writing the goal first while
        torque is still off, the servo holds in place until torque enable, at which
        point it's already pointing at the correct target.
        """
        follower = self._read_follower_joints()  # 6-dim (5 arm + 1 gripper)
        goal = {f"{n}": float(follower[i]) for i, n in enumerate(self._leader_motor_names)}
        try:
            self.leader.bus.sync_write("Goal_Position", goal)
        except Exception as exc:
            self._warn_leader_torque("主臂同步 follower 目标位姿失败", exc)
            return False
        return self._enable_leader_torque()

    def _hold_leader_at_pose(self, leader_target):
        """Hold the leader at its current operator-selected pose."""
        goal = {f"{n}": float(leader_target[i]) for i, n in enumerate(self._leader_motor_names)}
        try:
            self.leader.bus.sync_write("Goal_Position", goal)
        except Exception as exc:
            self._warn_leader_torque("主臂保持当前位置失败", exc)
            return False
        return self._enable_leader_torque()

    def _ee_error(self, leader, follower):
        leader_pose = self._station.get_ee_pose_from_joint(leader)
        follower_pose = self._station.get_ee_pose_from_joint(follower)
        return float(np.linalg.norm(leader_pose[:3] - follower_pose[:3]))

    def _leader_to_action(self, leader, follower):
        # Match lerobot-teleoperate for joint-mode takeover: send the leader's
        # current LeRobot motor-normalized joint targets directly to the follower.
        return leader.astype(np.float32, copy=True)

    def _ee_delta_action_from_motion(self, before_follower, after_follower, gripper_source):
        before_pose = self._station.get_ee_pose_from_joint(before_follower)
        after_pose = self._station.get_ee_pose_from_joint(after_follower)
        delta_m = after_pose[:3] - before_pose[:3]
        action = np.zeros((4,), dtype=np.float32)
        action[:3] = np.clip(delta_m / (self._ee_delta_scale + 1e-8), -1.0, 1.0)
        action[3] = 1.0 if gripper_source[-1] > self.gripper_binary_threshold else 0.0
        return action

    def _normalize_action_for_policy(self, action):
        """Convert executed physical SO101 action to policy training scale."""
        action = np.asarray(action, dtype=np.float32).copy()
        if self.control_mode == "pose":
            action[:3] = np.clip(action[:3], -1.0, 1.0)
            action[3] = 1.0 if action[3] >= 0.5 else 0.0
            return action
        action[:5] = 2.0 * (action[:5] - self._joint_min) / (self._joint_max - self._joint_min + 1e-8) - 1.0
        action[:5] = np.clip(action[:5], -1.0, 1.0)
        action[5] = 1.0 if action[5] > self.gripper_binary_threshold else 0.0
        return action

    def _record_leader_motion(self, leader):
        arm = leader[:-1].astype(np.float32, copy=True)
        motion = 0.0
        if self.prev_leader_arm is not None:
            motion = float(np.linalg.norm(arm - self.prev_leader_arm))
            self.leader_motion_queue.append(motion)
        self.prev_leader_arm = arm
        return motion

    def _manual_takeover_requested(self):
        return self.manual_takeover_enabled and bool(
            getattr(shared_state, "leader_manual_takeover", False)
        )

    def _start_intervention(self, arm_err, manual=False):
        self.is_intervening = True
        self.manual_intervention = bool(manual)
        self.intervention_started_at = time.perf_counter()
        self.leader_motion_queue.clear()
        self._disable_leader_torque(force=True)
        # During human takeover we execute raw leader joint targets, matching native
        # LeRobot teleoperation. Bypass the per-tick policy safety cap so the follower
        # can actually catch up to the leader; restore it before policy control resumes.
        if self._saved_max_relative_target is None:
            self._saved_max_relative_target = self._follower.config.max_relative_target
            self._follower.config.max_relative_target = None
        logging.info(
            "[SO101LeaderIntervention] takeover started (%s): leader/follower arm error %.4f %s",
            "manual" if self.manual_intervention else "auto",
            arm_err,
            "m" if self.control_mode == "pose" else "calibrated units",
        )
        print(
            f"[接管] {'手动' if self.manual_intervention else '自动'}接管已开启：从臂跟随主臂。",
            flush=True,
        )

    def _active_arm_error(self, joint_err, ee_err):
        return ee_err if self.control_mode == "pose" else joint_err

    def _should_start_intervention(self, joint_err, ee_err, manual_requested=False):
        if manual_requested:
            return True
        if not self.auto_takeover_enabled:
            return False
        if time.perf_counter() < self.release_guard_until:
            return False
        if self.control_mode == "pose":
            return ee_err is not None and ee_err > self.ee_error_threshold
        return joint_err > self.error_threshold

    def _should_release(self, arm_err, manual_requested=False):
        if not self.is_intervening or self.intervention_started_at is None:
            return False
        if self.manual_intervention:
            return not manual_requested
        if time.perf_counter() - self.intervention_started_at < self.min_intervention_s:
            return False
        if len(self.leader_motion_queue) < self.leader_motion_queue.maxlen:
            return False
        release_threshold = (
            self.ee_release_error_threshold
            if self.control_mode == "pose"
            else self.release_error_threshold
        )
        return (
            arm_err <= release_threshold
            and max(self.leader_motion_queue) <= self.release_motion_threshold
        )

    def _restore_policy_joint_cap(self):
        if self._saved_max_relative_target is not None:
            self._follower.config.max_relative_target = self._saved_max_relative_target
            self._saved_max_relative_target = None

    def _start_release_settle(self, leader_target, arm_err):
        self.is_intervening = False
        self.manual_intervention = False
        self.intervention_started_at = None
        self.leader_motion_queue.clear()
        self.is_release_settling = True
        self.release_settle_target = np.asarray(leader_target, dtype=np.float32).copy()
        self.release_settle_started_at = time.perf_counter()
        leader_held = self._hold_leader_at_pose(self.release_settle_target)
        if getattr(shared_state, "leader_manual_takeover", False):
            shared_state.leader_manual_takeover = False
        logging.info(
            "[SO101LeaderIntervention] manual takeover release settling: leader/follower arm error %.4f %s",
            arm_err,
            "m" if self.control_mode == "pose" else "calibrated units",
        )
        if leader_held:
            print("[接管] 手动接管结束：主臂保持当前位置，从臂正在追到该位姿。", flush=True)
        else:
            print("[接管] 手动接管结束：主臂保持失败；从臂仍会追到当前主臂位姿。", flush=True)

    def _finish_intervention(self, arm_err):
        self.is_intervening = False
        self.manual_intervention = False
        self.is_release_settling = False
        self.release_settle_target = None
        self.release_settle_started_at = None
        self.intervention_started_at = None
        self.leader_motion_queue.clear()
        self._restore_policy_joint_cap()
        self.release_guard_until = time.perf_counter() + self.release_guard_s
        if getattr(shared_state, "leader_manual_takeover", False):
            shared_state.leader_manual_takeover = False
        logging.info(
            "[SO101LeaderIntervention] takeover released: leader/follower arm error %.4f %s",
            arm_err,
            "m" if self.control_mode == "pose" else "calibrated units",
        )
        self._mirror_leader_to_follower()
        print("[接管] 从臂已追上主臂，恢复 policy 自主控制。", flush=True)

    def _release_leader_for_operator(self):
        """Leave the leader freely movable while the operator resets the scene."""
        if not self.leader_torque_enabled:
            return
        was_faulted = self.leader_torque_faulted
        ok = self._disable_leader_torque(force=True)
        if ok:
            return
        if self.leader_torque_faulted and not was_faulted:
            print(
                "[警告] 主臂扭矩释放失败，已停止主动镜像；请先让主臂卸力/断电重启再继续。",
                flush=True,
            )

    def _step_release_settle(self):
        target = self.release_settle_target
        if target is None:
            self._finish_intervention(0.0)
            return self.env.step({"action": self._read_follower_joints(), "so101_action_units": "raw", "so101_action_mode": "joint"})

        before_follower = self._read_follower_joints()
        obs, rew, terminated, truncated, info = self.env.step(
            {"action": target, "so101_action_units": "raw", "so101_action_mode": "joint"}
        )
        post_follower = self._read_follower_joints()
        post_joint_err = float(np.linalg.norm(target[:-1] - post_follower[:-1]))
        post_ee_err = self._ee_error(target, post_follower) if self.control_mode == "pose" else None
        post_arm_err = self._active_arm_error(post_joint_err, post_ee_err)
        elapsed = time.perf_counter() - (self.release_settle_started_at or time.perf_counter())
        release_threshold = (
            self.ee_release_error_threshold
            if self.control_mode == "pose"
            else self.release_error_threshold
        )
        if post_arm_err <= release_threshold or elapsed >= self.release_settle_timeout_s:
            self._finish_intervention(post_arm_err)
        if self.control_mode == "pose":
            intervene_action = self._ee_delta_action_from_motion(
                before_follower=before_follower,
                after_follower=post_follower,
                gripper_source=target,
            )
        else:
            intervene_action = self._normalize_action_for_policy(target)
        info["intervene_action"] = intervene_action
        info["is_intervention"] = True
        info["manual_takeover"] = False
        info["release_settle"] = True
        info["leader_follower_arm_error_deg"] = post_joint_err
        if post_ee_err is not None:
            info["leader_follower_ee_error_m"] = post_ee_err
        info["leader_motion_deg"] = 0.0
        return obs, rew, terminated, truncated, info

    def step(self, action):
        if self.is_release_settling:
            return self._step_release_settle()

        leader = self._read_leader_joints()
        follower = self._read_follower_joints()
        joint_err = float(np.linalg.norm(leader[:-1] - follower[:-1]))
        ee_err = self._ee_error(leader, follower) if self.control_mode == "pose" else None
        arm_err = self._active_arm_error(joint_err, ee_err)
        leader_motion = self._record_leader_motion(leader)
        manual_requested = self._manual_takeover_requested()

        if self.is_intervening and manual_requested:
            self.manual_intervention = True

        if not self.is_intervening and self._should_start_intervention(
            joint_err, ee_err, manual_requested=manual_requested
        ):
            self._start_intervention(arm_err, manual=manual_requested)

        if self.is_intervening:
            leader_action = self._leader_to_action(leader, follower)
            new_action = {
                "action": leader_action,
                "so101_action_units": "raw",
                "so101_action_mode": "joint",
            }
            obs, rew, terminated, truncated, info = self.env.step(new_action)
            post_follower = self._read_follower_joints()
            post_joint_err = float(np.linalg.norm(leader[:-1] - post_follower[:-1]))
            post_ee_err = self._ee_error(leader, post_follower) if self.control_mode == "pose" else None
            post_arm_err = self._active_arm_error(post_joint_err, post_ee_err)
            if self._should_release(post_arm_err, manual_requested=manual_requested):
                if self.manual_intervention:
                    self._start_release_settle(leader, post_arm_err)
                else:
                    self._finish_intervention(post_arm_err)
            if self.control_mode == "pose":
                intervene_action = self._ee_delta_action_from_motion(
                    before_follower=follower,
                    after_follower=post_follower,
                    gripper_source=leader,
                )
            else:
                intervene_action = self._normalize_action_for_policy(leader_action)
            info["intervene_action"] = intervene_action
            info["is_intervention"] = True
            info["manual_takeover"] = bool(self.manual_intervention)
            info["leader_follower_arm_error_deg"] = post_joint_err
            if post_ee_err is not None:
                info["leader_follower_ee_error_m"] = post_ee_err
            info["leader_motion_deg"] = leader_motion
            return obs, rew, terminated, truncated, info

        obs, rew, terminated, truncated, info = self.env.step(action)
        self._mirror_leader_to_follower()
        info["is_intervention"] = False
        info["manual_takeover"] = False
        info["leader_follower_arm_error_deg"] = joint_err
        if ee_err is not None:
            info["leader_follower_ee_error_m"] = ee_err
        info["leader_motion_deg"] = leader_motion
        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        # If the previous episode terminated mid-intervention, _finish_intervention
        # never ran and the per-tick cap is still bypassed. Restore it before the
        # upcoming reset/policy phase regains the safety throttle.
        if self._saved_max_relative_target is not None:
            self._follower.config.max_relative_target = self._saved_max_relative_target
            self._saved_max_relative_target = None
        self.manual_intervention = False
        self.is_release_settling = False
        self.release_settle_target = None
        self.release_settle_started_at = None
        shared_state.leader_manual_takeover = False
        self._release_leader_for_operator()

        base = self.env.unwrapped

        # Default to follower's current pose so if the user skips A and goes
        # straight to Space, env.reset's go_to_reset below is a no-op (no surprise
        # physical motion before the operator has positioned the leader).
        follower_now = self._read_follower_joints()
        base._reset_joint = follower_now[: base.joint_dim].astype(np.float32, copy=True)
        base.last_gripper_value = float(np.clip(follower_now[base.joint_dim], 0.0, 100.0))
        base.last_gripper_units = "raw"

        # ── Interactive alignment phase ──
        # User positions the leader. A → follower goes to leader's current 6-DoF
        # pose. Repeat A to refine. Space → start episode.
        print(
            "[环境重置] 调整主臂到 episode 初始位姿后：\n"
            "  A     = 从臂跟随主臂当前 6-DoF 位姿（可反复按 A 微调）\n"
            "  Space = 开始录制",
            flush=True,
        )
        shared_state.terminate = False
        shared_state.align_request = False
        while not shared_state.terminate:
            if shared_state.align_request:
                shared_state.align_request = False
                leader = self._read_leader_joints()
                base._reset_joint = leader[: base.joint_dim].astype(np.float32, copy=True)
                base.last_gripper_value = float(np.clip(leader[base.joint_dim], 0.0, 100.0))
                base.last_gripper_units = "raw"
                print("[对齐] 读取主臂位姿，从臂跟随中...", flush=True)
                base.go_to_reset(joint_reset=True)
                print("[对齐] 完成。再按 A 微调，或按 Space 开始。", flush=True)
            time.sleep(0.05)

        # Sync target to follower's now-aligned pose so the env.reset() below
        # finds go_to_reset a no-op (and fresh obs through the full wrapper chain
        # reflects the aligned physical state).
        follower_aligned = self._read_follower_joints()
        base._reset_joint = follower_aligned[: base.joint_dim].astype(np.float32, copy=True)
        base.last_gripper_value = float(np.clip(follower_aligned[base.joint_dim], 0.0, 100.0))
        base.last_gripper_units = "raw"
        base._skip_next_start_wait = True
        shared_state.terminate = False
        obs, info = self.env.reset(**kwargs)

        self.is_intervening = False
        self.manual_intervention = False
        self.intervention_started_at = None
        self.leader_motion_queue.clear()
        leader = self._read_leader_joints()
        follower = self._read_follower_joints()
        self.prev_leader_arm = leader[:-1].astype(np.float32, copy=True)
        joint_err = float(np.linalg.norm(leader[:-1] - follower[:-1]))
        ee_err = self._ee_error(leader, follower) if self.control_mode == "pose" else None
        arm_err = self._active_arm_error(joint_err, ee_err)
        if self._should_start_intervention(joint_err, ee_err, manual_requested=False):
            self._start_intervention(arm_err, manual=False)
        else:
            self._release_leader_for_operator()
        info["is_intervention"] = self.is_intervening
        info["manual_takeover"] = self.manual_intervention
        info["leader_follower_arm_error_deg"] = joint_err
        if ee_err is not None:
            info["leader_follower_ee_error_m"] = ee_err
        return obs, info

    def close(self):
        try:
            self._disable_leader_torque(force=True)
        finally:
            if self.leader is not None:
                try:
                    self.leader.disconnect()
                except Exception:
                    pass
        return self.env.close()


class AugmentedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.env = env

    def observation(self, obs):
        images = obs['images']
        env = self.env.unwrapped
        for key, img in images.items():
            if hasattr(env, 'image_crop'):
                cropped_rgb = env.image_crop[key](img) if key in env.image_crop else img
            else:
                cropped_rgb = img
            cropped_rgb = cv2.resize(
                cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
            )
            images[key] = cropped_rgb

        return obs
    
    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info






class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], Rotation.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )


        return observation


from collections import OrderedDict


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None, use_force=False):
        super().__init__(env)
        if proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())
        else:
            self.proprio_keys = list(proprio_keys)

        print("proprio_keys:", self.proprio_keys)    

        self.proprio_space = gym.spaces.Dict(
            OrderedDict((key, self.env.observation_space["state"][key]) for key in self.proprio_keys)
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        from collections import OrderedDict
        obs = {
            "state": flatten(
                self.proprio_space,
                OrderedDict((key, obs["state"][key]) for key in self.proprio_keys),
            ),
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

  
def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs
