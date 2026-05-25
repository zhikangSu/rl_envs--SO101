import time
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

    Two modes (auto-switched based on leader-follower joint position error):

    1. Non-intervention (policy autonomous):
       - leader.Torque_Enable = 1 (active servo)
       - Every step: read follower joint state, write to leader Goal_Position
       - Result: leader physically mirrors follower so when the human grabs the leader,
         it is already at the correct pose and there is no jump.

    2. Intervention (human takes over):
       - leader.Torque_Enable = 0 (passive, can be moved by hand)
       - Every step: read leader joint state, replace action with it
       - follower copies leader joint position (1:1, no IK in joint mode)

    Switch trigger: ||leader_arm_joints - follower_arm_joints||_2 > error_threshold
    (excludes gripper since gripper jitter is unrelated to "user is grabbing the arm").

    Reference: lerobot fork's BaseLeaderControlWrapper in
    src/lerobot/scripts/rl/gym_manipulator.py (EE-delta variant for SO100). This is the
    joint-mode simplification.
    """

    def __init__(self, env, error_threshold_deg=8.0, gripper_binary_threshold_pct=15.0):
        super().__init__(env)
        self.error_threshold = float(error_threshold_deg)
        self.gripper_binary_threshold = float(gripper_binary_threshold_pct)
        self.leader = None
        self._leader_motor_names = None
        self.leader_torque_enabled = False
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

    def step(self, action):
        leader = self._read_leader_joints()
        follower = self._read_follower_joints()
        arm_err = float(np.linalg.norm(leader[:-1] - follower[:-1]))

        # Pure-teleop mode: leader is always a passive position sensor (torque off
        # permanently), follower always tracks leader. This drops the original
        # HIL-SERL "policy autonomous / human grab to override" state machine because
        # for demonstration collection / dry-run there is no autonomous policy to
        # grab from. The torque-on mirror mode produced bad UX (had to push past the
        # error_threshold to overcome servo holding torque every cycle).
        #
        # If you later want HIL-SERL torque-assist (leader physically mirrors the
        # policy's commanded follower pose so the operator feels the agent's intent
        # and can grab to correct), revert this block to the original threshold-based
        # is_intervention logic and re-enable _mirror_leader_to_follower.
        self._disable_leader_torque()
        arm_target = leader[:-1]
        gripper_binary = 1.0 if leader[-1] > self.gripper_binary_threshold else 0.0
        new_action = np.concatenate([arm_target, [gripper_binary]]).astype(np.float32)
        obs, rew, terminated, truncated, info = self.env.step(new_action)
        info["intervene_action"] = new_action
        info["is_intervention"] = True
        info["leader_follower_arm_error_deg"] = arm_err
        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Keep leader passive across the reset as well — no mirror sync. Operator is
        # expected to hold the leader near the follower's reset pose before resuming.
        self._disable_leader_torque()
        info["is_intervention"] = True
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