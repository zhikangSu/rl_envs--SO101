"""Gym Interface for Franka、UR and Tienkung"""
import os
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import gymnasium as gym
import zmq
import pickle
import cv2
import copy     
from scipy.spatial.transform import Rotation
import time
from typing import Dict, Tuple, Optional, Union, Any, List
from pathlib import Path


# xrocs is optional — only required for franka/ur/tienkung. SO101 uses
# rl_envs.so101_station.SO101Station instead, which talks to lerobot.SO101Follower
# directly without any xrocs dependency.
try:
    from rl_envs.xrocs.core.config_loader import ConfigLoader
    from rl_envs.xrocs.core.station_loader import StationLoader
    _XROCS_AVAILABLE = True
except ImportError as _xrocs_err:
    ConfigLoader = None  # type: ignore
    StationLoader = None  # type: ignore
    _XROCS_AVAILABLE = False
    _XROCS_IMPORT_ERROR = _xrocs_err


##############################################################################
import traceback
import sys
from rl_envs.shared_state import shared_state


def decoder_image(camera_rgb_images, camera_depth_images, bgr2rgb=False):
    if type(camera_rgb_images[0]) is np.uint8:
        rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
        if bgr2rgb and rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if camera_depth_images is not None:
            depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
            depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        else:
            depth = np.asarray([])
        return rgb, depth
    else:
        rgb_images = []
        depth_images = []
        for idx, camera_rgb_image in enumerate(camera_rgb_images):
            rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            
            if bgr2rgb and rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb)
            depth_images.append(depth)
        rgb_images = np.asarray(rgb_images)
        depth_images = np.asarray(depth_images)
        return rgb_images, depth_images

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


class BaseEnv(gym.Env):
    def __init__(
        self,
        fake_env=False,
        config=None,
    ):
        self.config = config
        self._gripper_sleep = config.gripper_sleep
        self.joint_dim = config.joint_dim
        self.dual_arm = config.dual_arm

        if self.dual_arm:
            self._reset_joint = {name: np.array(config.reset_joint[name])[0:self.joint_dim] for name in config.reset_joint.keys()}
        else:
            self._reset_joint = np.array(config.reset_joint)[0:self.joint_dim]
        self._random_xy_range = config.random_xy_range
        self._random_rz_range = config.random_rz_range
        self._random_reset = config.random_reset
        self._bgr2rgb = config.bgr2rgb
        self._image_keys = config.image_keys
        self.robot_type = config.robot_type
        self.action_scale = config.action_scale
        self.max_episode_length = config.max_episode_length
        self.control_mode = config.control_mode
        self.close_gripper = config.close_gripper
        self.fix_gripper = config.fix_gripper
        self.ego_mode = config.ego_mode
        self.use_cmd_pose = config.use_cmd_pose
        self.last_gripper_units = "policy"
        
        assert self.control_mode in ["joint", "pose"], f'Not valid control mode: {self.control_mode}'
        
        self.fake_env = fake_env
        self.hz = config.hz

        image_resize = config.image_resize
        tcp_pose_dim = 7  # 末端执行器位姿：xyz 位置(3) + 四元数旋转(4) = 7维
        if self.dual_arm:
            tcp_pose_dim = 2 * tcp_pose_dim  # 双臂，每臂7维
        
        gripper_dim = 1
        if self.dual_arm:
            gripper_dim = 2 * gripper_dim
        
        joint_dim = self.joint_dim
        if self.dual_arm:
            joint_dim = 2 * joint_dim
        
        state_dict = {
            "tcp_pose": gym.spaces.Box(
                -np.inf, np.inf, shape=(tcp_pose_dim,)
            ),  
            "gripper_pose": gym.spaces.Box(0, 1, shape=(gripper_dim,)),  # 夹爪开合度：0=完全打开，1=完全关闭
            "joints": gym.spaces.Box(
                -np.inf, np.inf, shape=(joint_dim,)
            ),
        }

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_dict),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=image_resize[key], dtype=np.uint8) 
                        for key in config.image_keys}
                ),
            }
        )

    
        if self.control_mode == "pose":
            if self.dual_arm:
                # action_space = left(xyz + rpy + gripper) + right(xyz + rpy + gripper)
                self.action_space = gym.spaces.Box( 
                    np.array([-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                )
            else:
                # action_space = (xyz + rpy + gripper)
                self.action_space = gym.spaces.Box(
                    np.array([-1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                    np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                )
        else:
            self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(joint_dim + gripper_dim,))

        if fake_env:
            print_green("fake env : not connect to robot")
            return

        # ───── SO101: lerobot.SO101Follower (Feetech USB), no xrocs ─────
        if "so101" in self.robot_type:
            from rl_envs.so101_station import SO101Station
            self.robot_station = SO101Station(config)
            self.robot_station.connect()
            self.cfg_dict = None  # xrocs-only field, kept for compatibility
        else:
            # ───── franka / ur / tienkung: xrocs station ─────
            if not _XROCS_AVAILABLE:
                raise ImportError(
                    f"xrocs is not installed but robot_type={self.robot_type!r} requires it. "
                    f"Install xrocs (rl_envs/xrocs && pip install -e .) or switch to "
                    f"robot_type containing 'so101'. Original import error: {_XROCS_IMPORT_ERROR}"
                )
            base_dir = Path(__file__).parent.absolute()
            config_path = base_dir / "xrocs" / "configuration.toml"
            cfg_loader = ConfigLoader(str(config_path))
            self.cfg_dict = cfg_loader.get_config()
            station_loader = StationLoader(self.cfg_dict)
            self.robot_station = station_loader.generate_station_handle()
            self.robot_station.connect()

        if self.dual_arm:
            xyz_limits = {}

            for arm_name in config.abs_pose_limit_low.keys():
                low_vals = np.array(config.abs_pose_limit_low[arm_name])
                high_vals = np.array(config.abs_pose_limit_high[arm_name])

                # XYZ limits (前3维)
                xyz_limits[arm_name] = gym.spaces.Box(
                    low=low_vals[:3],
                    high=high_vals[:3],
                    dtype=np.float64
                )


            self.xyz_bounding_box = gym.spaces.Dict(xyz_limits)
        else:
            self.xyz_bounding_box = gym.spaces.Box(
                np.array(config.abs_pose_limit_low[:3]),
                np.array(config.abs_pose_limit_high[:3]),
                dtype=np.float64,
            )
        
        
        self.image_crop = {}

        if hasattr(config, 'image_crop') and config.image_crop is not None:
            for camera_key, crop_func in config.image_crop.items():
                if callable(crop_func):
                    self.image_crop[camera_key] = crop_func
                else:
                    def make_crop_func(crop_params):
                        crop_params = list(crop_params)
                        if isinstance(crop_params, (list, tuple)) and len(crop_params) == 2:
                            h_range, w_range = crop_params
                            return lambda img: img[h_range[0]:h_range[1], w_range[0]:w_range[1]]
                        else:
                            return lambda img: img
                    
                    self.image_crop[camera_key] = make_crop_func(crop_func)
          


    def clip_safety_box(self, pose: np.ndarray, arm_name: str=None) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        if self.dual_arm and arm_name is not None:
            xyz_bounding_box = self.xyz_bounding_box[arm_name]
            pose[:3] = np.clip(
                pose[:3], xyz_bounding_box.low, xyz_bounding_box.high
            )
        else:
            pose[:3] = np.clip(
                pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
            )
        return pose

    def pose_quat2euler(self, pose):
        pose_t, pose_quat = pose[0:3], pose[3:7]
        pose_euler = Rotation.from_quat(pose_quat).as_euler("xyz")
        pose = np.hstack([pose_t, pose_euler])
        return pose

    def pose_euler2quat(self, pose):
        pose_t, pose_euler = pose[0:3], pose[3:6]
        pose_quat = Rotation.from_euler("xyz", pose_euler).as_quat(canonical=True)
        pose = np.hstack([pose_t, pose_quat])
        return pose
    

    def get_xtele(self) -> dict:
        """
        获取遥操作设备状态
        """
        if "so101" in self.robot_type:
            # SO101 uses SO101LeaderIntervention wrapper (see rl_envs/wrappers.py)
            # which talks to the leader arm directly. The xtele path is unused.
            raise NotImplementedError(
                "SO101 intervention does not go through xtele. Use "
                "intervention_backend='leader_so101' in cfg/config.yaml."
            )
        if 'ur' in self.robot_type:
            self.tele_agent.switch_act()
            joints = self.tele_agent.act()
            joints = list(joints)
            xtele_ee_pose = self.robot_station.get_ee_pose_from_joint(joints[0:self.joint_dim])
            
            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }
            return recv 
        elif 'franka' in self.robot_type:
            self.tele_agent.exit_any_sync(0)
            joints = self.tele_agent.act()
            joints = list(joints)
            xtele_ee_pose = self.robot_station.get_ee_pose_from_joint(joints[0:self.joint_dim])
            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }
            return recv
        elif 'tienkung' in self.robot_type:
            self.tele_agent.exit_any_sync(0)
            joints = self.tele_agent.act()
            xtele_ee_pose = {}
            for name in joints.keys():
                xtele_ee_pose[name] = self.robot_station.get_ee_pose_from_joint(joints[name])
            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }
            return recv
        else:
            raise NotImplementedError("Unknown robot type")

    def init_xtele(self,):
        """
        初始化遥操作模块
        """
        if "so101" in self.robot_type:
            # SO101 leader is handled by SO101LeaderIntervention wrapper, not xtele.
            return
        if 'ur' in self.robot_type or 'franka' in self.robot_type or "tienkung" in self.robot_type:
            from xtele.core.integrate_module import TeleCore
            self.tele_agent = TeleCore()
        else:
            raise NotImplementedError("Unknown robot type")

    def sync_xtele(self, timeout: float = 5):
        """
        同步机器人状态到遥操作设备
        """
        if "so101" in self.robot_type:
            # SO101 leader→follower sync is handled inside SO101LeaderIntervention.step()
            # (writes follower joint readings to leader Goal_Position every tick).
            return

        if 'ur' in self.robot_type:
            goal = np.append(self.curr_arm_joints, self.curr_gripper_joints)

            self.tele_agent.switch_reverse()
            self.tele_agent.sync_position(goal)
        elif 'franka' in self.robot_type:
            goal = np.append(self.curr_arm_joints, self.curr_gripper_joints)
            
            need_torque = False
            tele_cur_joints = self.tele_agent.act()
            tele_tar_joints = goal
            timeout = int(timeout // 0.02)
            if timeout <= 1:
                path = np.array([tele_tar_joints])
            else:
                path = np.linspace(tele_cur_joints, tele_tar_joints, timeout)
            for p in path:
                self.tele_agent.sync_position_torque(p)
                time.sleep(0.02)

        elif "tienkung" in self.robot_type:
            goal = []
            
            for name in self.curr_arm_joints.keys():
                arm_joints = self.curr_arm_joints[name]
                hand_joints = self.curr_gripper_joints[name]

                goal = goal + list(arm_joints)
                goal.append(hand_joints)
            self.tele_agent.set_position(goal)
        else:
            raise NotImplementedError("Unknown robot type")    

    def compute_next_pose(self, curr_pose, delta_action) -> np.ndarray:
        '''
        input:
            curr_pose: np.ndarray, shape: (7,) x,y,z, quat_x, quat_y, quat_z, quat_w
            delta_action: np.ndarray, shape: (7,) x,y,z, r, p, y, gripper
        output:
            next_pose: np.ndarray, shape: (7,) x, y, z, r, p, y, gripper
        '''
        
        curr_pose_t, curr_pose_quat = curr_pose[0:3], curr_pose[3:7]
        curr_mat = np.eye(4)
        curr_mat[:3, :3] = Rotation.from_quat(curr_pose_quat).as_matrix()
        curr_mat[:3, 3] = curr_pose_t
        
        delta_t = delta_action[0:3] * self.action_scale[0]
        delta_euler = delta_action[3:6] * self.action_scale[1]
        delta_mat = np.eye(4)  # 4x4 齐次变换矩阵
        delta_mat[:3, :3] = Rotation.from_euler("xyz", delta_euler).as_matrix()  # 旋转矩阵
        delta_mat[:3, 3] = delta_t  # 平移部分
        
        tar_mat = curr_mat @ delta_mat
        
        tar_t = tar_mat[:3,3]
        tar_euler = Rotation.from_matrix(tar_mat[:3, :3]).as_euler("xyz")
        tar_gripper = delta_action[-1] * self.action_scale[2]
        
        next_pose = np.hstack([tar_t, tar_euler, tar_gripper])
        return next_pose
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], int, bool, bool, Dict[str, Any]]:
        # ========== 控制执行频率 ==========
        start_time = time.time()  # 记录开始时间
        action_units = None
        if isinstance(action, dict) and "action" in action:
            action_units = action.get("so101_action_units")
            action = np.asarray(action["action"], dtype=np.float32)
        elif "so101" in self.robot_type and self.control_mode == "joint":
            # Actor actions are SAC tanh outputs in [-1, 1]. Reset paths bypass
            # BaseEnv.step() and call _send_joint_command() directly with raw
            # LeRobot motor-normalized joint targets.
            action_units = "policy"
        
        # Policy gripper commands keep the original sleep guard. Raw SO101
        # takeover should match lerobot-teleoperate, so the leader gripper is
        # forwarded continuously unless the environment explicitly fixes it.
        if action_units == "raw" and "so101" in self.robot_type and self.control_mode == "joint":
            include_gripper = not self.fix_gripper
        elif (time.time() - self.last_gripper_act > self._gripper_sleep) and not self.fix_gripper:
            include_gripper = True
        else:
            include_gripper = False

        if self.control_mode == "joint":
            self._send_joint_command(action, include_gripper, action_units=action_units)
            # curr_pose_euler = None
        elif self.control_mode == "pose":
            action = action.clip(-1, 1)
            if self.dual_arm:
                if "tienkung" in self.robot_type:
                    next_pose = {}
                    dual_arm_action = {
                        "left": action[0:7],
                        "right": action[7:14],
                    }

                    for name, action in dual_arm_action.items():
                        curr_pose = self.currpos[name]
                        next_pose[name] = self.compute_next_pose(curr_pose, action)

                        # 裁剪到安全边界
                        next_pose[name] = self.clip_safety_box(next_pose[name], arm_name=name)  

                    self.currpos = {name: self.pose_euler2quat(pose) for name, pose in next_pose.items()}
                else:
                    raise NotImplementedError("Unknown robot type")
                
            
            else:
                next_pose = self.compute_next_pose(self.currpos, action)
                next_pose = self.clip_safety_box(next_pose)
                self.currpos = self.pose_euler2quat(next_pose)

            self._send_pos_command(next_pose, include_gripper) 

            # curr_pose_euler = self.pose_quat2euler(obs['arm_pose']['single'])
        else:
            raise NotImplementedError(f"Not valid control mode: {self.control_mode}")

        if include_gripper:
            self.last_gripper_act = time.time()


        self.curr_path_length += 1

        end_time = time.time()
        sleep_time = max(0, 1/self.hz - (end_time - start_time))
        time.sleep(sleep_time)

        if self.use_cmd_pose:
            obs = self._get_obs(update=False)
        else:
            obs = self._get_obs()
        
        reward = 0.0
        terminated = False
        truncated = self.curr_path_length >= self.max_episode_length
        return obs, int(reward), terminated, truncated, {"succeed": terminated, "is_intervention": False}


    def reset(self, **kwargs):
        self._update_currpos()
        self.last_gripper_act = time.time()
        if self.dual_arm:
            self.last_gripper_value = {
                "left": 1.0 if self.close_gripper else 0.0,
                "right": 1.0 if self.close_gripper else 0.0,
            }
        else:
            self.last_gripper_value = 1.0 if self.close_gripper else 0.0
        self.last_gripper_units = "policy"

        if self.ego_mode:
            
            self.sync_xtele()
            while True:
                try:
                    input("[等待确认] 按 Enter 继续...")
                    break  # Break the loop if input is successful
                except (EOFError, ValueError):
                    print("[提示] 暂时无法读取输入，正在重试...", flush=True)
                    traceback.print_exc()
                    time.sleep(1)
                    continue  

            shared_state.terminate = False
            print("[等待开始] 请重置场景，确认安全后按 Space 开始。", flush=True)
            while not shared_state.terminate:
                obs = self.get_xtele()
                xtele_joints = obs['joints']
                self._update_currpos()
                target_joint = xtele_joints.copy()
                action_units = "raw" if "so101" in self.robot_type else None
                self._send_joint_command(target_joint, include_gripper=True, action_units=action_units)
                time.sleep(1 / self.hz)

        print("\n[环境重置] 从臂正在回到 reset 位姿，请注意避让。", flush=True)
        self.go_to_reset(joint_reset=True)      
        shared_state.terminate = False
        print("[等待开始] 请摆好方块和杯子；确认安全后按 Space 开始下一条录制。", flush=True)
        while not shared_state.terminate:
            continue
        shared_state.terminate = False

        self.curr_path_length = 0
        self.last_gripper_act = time.time()
        if self.dual_arm:
            self.last_gripper_value = {
                "left": 1.0 if self.close_gripper else 0.0,
                "right": 1.0 if self.close_gripper else 0.0,
            }
        else:
            self.last_gripper_value = 1.0 if self.close_gripper else 0.0
        self.last_gripper_units = "policy"
        obs = self._get_obs()
        return obs, {"succeed": False, "is_intervention": False}


    def go_to_reset(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """        
        # perform joint reset if needed
        self._update_currpos()
        if self.dual_arm:
            if "tienkung" in self.robot_type:
                path = {}
                for name, joint in self._reset_joint.items():
                    goal_joint = joint.copy()
                    assert len(goal_joint) == self.joint_dim

                    curr_joint = np.concatenate([
                        self.curr_arm_joints[name], np.array(self.curr_gripper_joints[name])
                    ])
                    goal_joint = np.concatenate([goal_joint, np.array([self.last_gripper_value[name]])])
                    cnt = int(3 / (1 / self.hz))
                    path[name] = np.linspace(curr_joint, goal_joint, cnt)

                for j_left, j_right in zip(path["left"], path["right"]):
                    joint_cmd = {
                            "left": j_left,
                            "right": j_right
                        }
                    self._send_joint_command(joint_cmd, include_gripper=False)
                    time.sleep(1 / self.hz)

                self._update_currpos()
                curr_pose = self.currpos.copy()

                # If random reset is enabled, add random perturbations to the xy plane and rotation angle
                if self._random_reset:
                    reset_pose = {}
                    for name, pose in curr_pose.items():
                        single_reset_pose = self.pose_quat2euler(pose).copy()
                        # 在xy平面上添加随机偏移
                        single_reset_pose[:2] += np.random.uniform(
                            -self._random_xy_range, self._random_xy_range, (2,)
                        )
                        # 获取旋转角
                        axis_random = np.array(single_reset_pose[3:])
                        assert axis_random.shape == (3,)
                        # 在Z轴旋转角上添加随机扰动
                        axis_random[-1] += np.random.uniform(
                            -self._random_rz_range, self._random_rz_range
                        )
                        single_reset_pose[3:] = axis_random
                        reset_pose[name] = single_reset_pose
                        
                    self._send_pos_command(reset_pose, include_gripper=False)
                    time.sleep(1 / self.hz)   
            
            else:
                raise NotImplementedError("Unknown robot type")
        else:

            curr_pose = self.currpos.copy()
            curr_pose = self.pose_quat2euler(curr_pose)

            assert self._reset_joint.shape == (self.joint_dim,)
            
            goal_joints = self._reset_joint.copy()
            # Combine the current joints and gripper position 
            curr_joints = np.concatenate([self.curr_arm_joints, np.array([self.curr_gripper_joints])])
            # Combine the target joints and gripper position
            goal_joints = np.concatenate([goal_joints, np.array([self.last_gripper_value])])
            cnt = int(3 / (1 / self.hz))
            # Generate a linear interpolation path from the current joints to the target joints (smooth transition)
            path = np.linspace(curr_joints, goal_joints, cnt)
            for p in path:
                action_units = "raw" if "so101" in self.robot_type else None
                self._send_joint_command(p, include_gripper=False, action_units=action_units)
                time.sleep(1 / self.hz)

            self._update_currpos()
            curr_pose = self.currpos.copy()

            # If random reset is enabled, add random perturbations to the xy plane and rotation angle
            if self._random_reset:
                reset_pose = self.pose_quat2euler(curr_pose).copy()
                # 在xy平面上添加随机偏移
                reset_pose[:2] += np.random.uniform(
                    -self._random_xy_range, self._random_xy_range, (2,)
                )
                # 获取旋转角
                axis_random = np.array(reset_pose[3:])
                assert axis_random.shape == (3,)
                # 在Z轴旋转角上添加随机扰动
                axis_random[-1] += np.random.uniform(
                    -self._random_rz_range, self._random_rz_range
                )
                reset_pose[3:] = axis_random

                self._send_pos_command(reset_pose, include_gripper=False)
                time.sleep(1 / self.hz)


    def _send_joint_command(self, joints: np.ndarray, include_gripper=False, action_units=None):
        if self.dual_arm:
            '''
            joints = {
                "left": np.zeros(joint_dim+gripper_dim),
                "right": np.zeros(joint_dim+gripper_dim),
            }
            '''
            if "tienkung" in self.robot_type:
                robot_target = {
                    "arm": {
                        "position": {}
                    }
                }
                for name in joints.keys():
                    gripper_value = joints[name][-1] if include_gripper else self.last_gripper_value[name]
                    gripper_value_binary = 1.0 if gripper_value >= 0.5 else 0.0
                    if include_gripper:
                        self.last_gripper_value[name] = gripper_value_binary
                    robot_target["arm"]["position"][name] = np.append(joints[name][0:self.joint_dim], self.last_gripper_value[name])
                        
                obs = self.robot_station.step(robot_target)
            else:
                raise NotImplementedError("Unknown robot type")
        else:
            """
                joints = np.zeros(joint_dim+gripper_dim)
            """
            gripper_value = joints[-1] if include_gripper else self.last_gripper_value
            gripper_value_binary = 1.0 if gripper_value >= 0.5 else 0.0
            if include_gripper:
                self.last_gripper_value = gripper_value_binary
            if "ur" in self.robot_type:
                robot_target = {
                    "arm_joints": {
                        "single": joints[0:self.joint_dim]
                    },
                    "hand_joints": {"single": self.last_gripper_value}
                }
                obs = self.robot_station.step(robot_target)
                return obs
            elif "franka" in self.robot_type:
                robot_target = {
                    "arm": {
                        "position": {
                            "single": np.append(joints[0:self.joint_dim], self.last_gripper_value),
                        }
                    }
                }
                obs = self.robot_station.step(robot_target)
            elif "so101" in self.robot_type:
                # SO101 joint mode: policy actions are [-1, 1] arm targets +
                # binary gripper; leader takeover actions are raw LeRobot
                # motor-normalized values, matching lerobot-teleoperate.
                gripper_units = self.last_gripper_units
                if include_gripper and action_units == "raw":
                    self.last_gripper_value = float(np.clip(gripper_value, 0.0, 100.0))
                    self.last_gripper_units = "raw"
                    gripper_units = "raw"
                elif include_gripper:
                    self.last_gripper_value = gripper_value_binary
                    self.last_gripper_units = "policy"
                    gripper_units = "policy"
                robot_target = {
                    "arm_joints": {
                        "single": joints[0:self.joint_dim]
                    },
                    "hand_joints": {"single": self.last_gripper_value},
                    "so101_action_units": action_units,
                    "so101_gripper_units": gripper_units,
                }
                obs = self.robot_station.step(robot_target)
                return obs
            else:
                raise NotImplementedError("Unknown robot type")


    def _send_pos_command(self, pose: np.ndarray, include_gripper=False):
        if self.dual_arm:
            '''
            pose = {
                "left": np.zeros(6 + gripper_dim),
                "right": np.zeros(6 + gripper_dim),
            }
            '''
            if "tienkung" in self.robot_type:
                robot_target = {
                    "arm_pose": {},
                    "hand_joints": {}
                }
                for name in pose.keys():
                    gripper_value = pose[name][-1] if include_gripper else self.last_gripper_value[name]
                    gripper_value_binary = 1.0 if gripper_value >= 0.5 else 0.0
                    robot_target["arm_pose"][name] = pose[name]
                    robot_target["hand_joints"][name] = self.last_gripper_value[name]
                obs = self.robot_station.step_ee(robot_target)
            else:
                raise NotImplementedError("Unknown robot type")
        else:
            """
            pose = np.zeros(6 + gripper_dim)
            """
            if include_gripper:
                gripper_value_binary = 1.0 if pose[-1] >= 0.5 else 0.0
                self.last_gripper_value = gripper_value_binary

            if "ur" in self.robot_type:
                robot_target = {
                    "arm_pose": {
                        "single": pose[0:6]
                    },
                    "hand_joints": {"single": self.last_gripper_value}
                }
                obs = self.robot_station.step_ee(robot_target)
            elif "franka" in self.robot_type:
                arm_pose = np.append(pose[0:6], self.last_gripper_value)
                robot_target = {
                    "arm_pose": {
                        "single": arm_pose
                    },
                    "hand_joints": {}
                }
                obs = self.robot_station.step_ee(robot_target)
            else:
                raise NotImplementedError("Unknown robot type")


    def _update_currpos(self, obs=None, update=True):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        if obs is None:
            obs = self._get_obs_from_robot()
        if self.dual_arm:
            if "tienkung" in self.robot_type:
                if update:
                    self.currpos = obs["arm_pose"]
                self.curr_gripper_joints = obs["hand_joints"]
                self.curr_arm_joints = obs["arm_joints"]
                
            else:
                raise NotImplementedError("Unknown robot type")
        else:
            if update:
                self.currpos = obs["arm_pose"]['single']
            self.curr_gripper_joints = np.array(obs["hand_joints"]['single']).squeeze()
            self.curr_arm_joints = np.array(obs["arm_joints"]['single'][0:self.joint_dim])
        return obs

    def _get_obs(self, obs=None, update=True) -> dict:
        if self.fake_env:
            images = self.get_fake_im()
            state_observation = self.get_fake_pose()
        else:
            obs = self._update_currpos(obs, update)

            images = {}
            for key, cap in obs["images"].items():
                if key not in self._image_keys:
                    continue
                # SO101 path: SO101Station.get_obs() already returns RGB uint8 ndarrays
                # (H, W, 3) from lerobot OpenCVCamera. xrocs station returns JPEG bytes
                # that need cv2.imdecode via decoder_image().
                if "so101" in self.robot_type:
                    rgb = cap
                    if hasattr(rgb, "numpy"):
                        rgb = rgb.numpy()
                    if rgb.dtype != np.uint8:
                        rgb = rgb.astype(np.uint8)
                else:
                    rgb, _ = decoder_image(cap, None, bgr2rgb=self._bgr2rgb)
                images[key] = rgb
                if not os.path.exists(f"online_image_{key}.png"):
                    cv2.imwrite(f"online_image_{key}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if self.dual_arm:
                if "tienkung" in self.robot_type:
                    state_observation = {
                        "tcp_pose": self._flatten(self.currpos),
                        "gripper_pose": self._flatten(self.curr_gripper_joints),
                        "joints": self._flatten(self.curr_arm_joints),
                    }

                else:
                    raise NotImplementedError("Unknown robot type") 
            else:
                state_observation = {
                    "tcp_pose": self.currpos,
                    "gripper_pose": self.curr_gripper_joints,
                    "joints": self.curr_arm_joints,
                }
            
        return dict(images=images, state=state_observation)


    def _get_obs_from_robot(self) -> dict:
        obs = self.robot_station.get_obs()

        # SO101 joint mode: SO101Station already returns the identity arm_pose 7-vec
        # (no FK), no quaternion canonicalization needed.
        if "so101" in self.robot_type:
            return obs

        if self.dual_arm:
            if "tienkung" in self.robot_type:
                arm_pose = obs["arm_pose"]
                arm_pose_t = {name: arm_pose[name][0:3] for name in arm_pose}
                arm_pose_quat = {name: arm_pose[name][3:] for name in arm_pose}
                arm_pose_quat = {name: Rotation.from_quat(arm_pose_quat[name]).as_quat(canonical=True) for name in arm_pose_quat}
                arm_pose = {name: np.hstack([arm_pose_t[name], arm_pose_quat[name]]) for name in arm_pose}
                obs["arm_pose"] = arm_pose
            else:
                raise NotImplementedError("Unknown robot type")
        else:
            arm_pose = obs['arm_pose']['single']
            arm_pose_t, arm_pose_quat = arm_pose[0:3], arm_pose[3:]
            arm_pose_quat = Rotation.from_quat(arm_pose_quat).as_quat(canonical=True)
            arm_pose = np.hstack([arm_pose_t, arm_pose_quat])
            obs['arm_pose'] = {'single': arm_pose}
        return obs

    def _flatten(self, input_dict: Dict[str, Any]) -> np.ndarray:
        flattened_values = []
        for key in input_dict.keys():
            value = input_dict[key]
            flattened_values.append(np.array(value).flatten())
        return np.concatenate(flattened_values)


    def close(self):
        return
