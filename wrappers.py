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
    """SO101 leader-arm intervention for joint control.

    Default mode is policy-autonomous: the policy command is sent to the follower, and
    the leader is servoed to mirror the follower's current joint state. If the operator
    pulls the leader away from the mirrored pose, the wrapper switches to intervention:
    leader torque is disabled and follower commands are replaced by leader joint targets.

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
    ):
        super().__init__(env)
        self.error_threshold = float(error_threshold_deg)
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
        cfg = self.env.unwrapped.config
        self._joint_min = np.asarray(list(cfg.so101_joint_action_min), dtype=np.float32)
        self._joint_max = np.asarray(list(cfg.so101_joint_action_max), dtype=np.float32)
        # Cache follower so we can bypass max_relative_target during human takeover.
        # The per-tick clamp is a safety net for the policy; while a human is driving
        # the leader, it just throttles the follower below leader hand speed and the
        # follower never reaches the leader's pose. Save the original cap and disable
        # it on _start_intervention, restore it on _finish_intervention.
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
                self.leader.bus.write("P_Coefficient", motor, 16)
                self.leader.bus.write("I_Coefficient", motor, 0)
                self.leader.bus.write("D_Coefficient", motor, 16)
            except Exception as exc:
                logging.warning(
                    "[SO101LeaderIntervention] failed to soften leader gains for %s: %s",
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

    def _enable_leader_torque(self):
        if not self.leader_torque_enabled:
            self.leader.bus.sync_write("Torque_Enable", 1)
            self.leader_torque_enabled = True

    def _disable_leader_torque(self):
        if self.leader_torque_enabled:
            self.leader.bus.sync_write("Torque_Enable", 0)
            self.leader_torque_enabled = False

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
        self.leader.bus.sync_write("Goal_Position", goal)
        self._enable_leader_torque()

    def _leader_to_action(self, leader):
        # Match lerobot-teleoperate for takeover: send the leader's current
        # LeRobot motor-normalized joint targets directly to the follower.
        return leader.astype(np.float32, copy=True)

    def _normalize_action_for_policy(self, action):
        """Convert executed physical SO101 action to policy training scale."""
        action = np.asarray(action, dtype=np.float32).copy()
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

    def _start_intervention(self, arm_err):
        self.is_intervening = True
        self.intervention_started_at = time.perf_counter()
        self.leader_motion_queue.clear()
        self._disable_leader_torque()
        # Bypass the per-tick relative-target cap so the follower can keep up
        # with leader hand motion (native servo speed ~300°/s vs. capped 60°/s).
        if self._saved_max_relative_target is None:
            self._saved_max_relative_target = self._follower.config.max_relative_target
            self._follower.config.max_relative_target = None
        logging.info(
            "[SO101LeaderIntervention] takeover started: leader/follower arm error %.2f calibrated units",
            arm_err,
        )

    def _should_release(self, arm_err):
        if not self.is_intervening or self.intervention_started_at is None:
            return False
        if time.perf_counter() - self.intervention_started_at < self.min_intervention_s:
            return False
        if len(self.leader_motion_queue) < self.leader_motion_queue.maxlen:
            return False
        return (
            arm_err <= self.release_error_threshold
            and max(self.leader_motion_queue) <= self.release_motion_threshold
        )

    def _finish_intervention(self, arm_err):
        self.is_intervening = False
        self.intervention_started_at = None
        self.leader_motion_queue.clear()
        # Restore the per-tick cap before handing control back to the policy.
        if self._saved_max_relative_target is not None:
            self._follower.config.max_relative_target = self._saved_max_relative_target
            self._saved_max_relative_target = None
        self._mirror_leader_to_follower()
        logging.info(
            "[SO101LeaderIntervention] takeover released: leader/follower arm error %.2f calibrated units",
            arm_err,
        )

    def step(self, action):
        leader = self._read_leader_joints()
        follower = self._read_follower_joints()
        arm_err = float(np.linalg.norm(leader[:-1] - follower[:-1]))
        leader_motion = self._record_leader_motion(leader)

        if not self.is_intervening and arm_err > self.error_threshold:
            self._start_intervention(arm_err)

        if self.is_intervening:
            new_action = {
                "action": self._leader_to_action(leader),
                "so101_action_units": "raw",
            }
            obs, rew, terminated, truncated, info = self.env.step(new_action)
            post_follower = self._read_follower_joints()
            post_arm_err = float(np.linalg.norm(leader[:-1] - post_follower[:-1]))
            if self._should_release(post_arm_err):
                self._finish_intervention(post_arm_err)
            info["intervene_action"] = self._normalize_action_for_policy(new_action["action"])
            info["is_intervention"] = True
            info["leader_follower_arm_error_deg"] = post_arm_err
            info["leader_motion_deg"] = leader_motion
            return obs, rew, terminated, truncated, info

        obs, rew, terminated, truncated, info = self.env.step(action)
        self._mirror_leader_to_follower()
        info["is_intervention"] = False
        info["leader_follower_arm_error_deg"] = arm_err
        info["leader_motion_deg"] = leader_motion
        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        # If the previous episode terminated mid-intervention, _finish_intervention
        # never ran and the per-tick cap is still bypassed. Restore it before the
        # upcoming reset/policy phase regains the safety throttle.
        if self._saved_max_relative_target is not None:
            self._follower.config.max_relative_target = self._saved_max_relative_target
            self._saved_max_relative_target = None

        # Align follower (arm + gripper) to leader's current 6-DoF pose so the
        # new episode starts with leader/follower matched — operator positions
        # the leader at the desired initial pose, follower lands at the same
        # pose. Avoids the spurious first-step takeover that fires when the
        # follower defaults to cfg.reset_joint but the leader is elsewhere.
        leader_initial = self._read_leader_joints()  # 6-dim (5 arm + 1 gripper)
        base = self.env.unwrapped
        base._reset_joint = leader_initial[: base.joint_dim].astype(np.float32, copy=True)
        base.last_gripper_value = float(np.clip(leader_initial[base.joint_dim], 0.0, 100.0))
        base.last_gripper_units = "raw"
        obs, info = self.env.reset(**kwargs)

        self.is_intervening = False
        self.intervention_started_at = None
        self.leader_motion_queue.clear()
        leader = self._read_leader_joints()
        follower = self._read_follower_joints()
        self.prev_leader_arm = leader[:-1].astype(np.float32, copy=True)
        arm_err = float(np.linalg.norm(leader[:-1] - follower[:-1]))
        if arm_err > self.error_threshold:
            self._start_intervention(arm_err)
        else:
            self._mirror_leader_to_follower()
        info["is_intervention"] = self.is_intervening
        info["leader_follower_arm_error_deg"] = arm_err
        return obs, info

    def close(self):
        try:
            self._disable_leader_torque()
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
        if use_force:
            self.proprio_keys = proprio_keys
        else:
            self.proprio_keys = proprio_keys[:2]

        print("proprio_keys:", self.proprio_keys)    

        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

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
