"""SO101 hardware adapter exposing xrocs-station-compatible interface.

`rl_envs.base_env.BaseEnv` expects a `self.robot_station` object with these methods:
  - connect()
  - step(robot_target) -> obs              (joint command)
  - step_ee(robot_target) -> obs           (EE-delta command via FK/IK)
  - get_obs() -> obs
  - get_ee_pose_from_joint(joint) -> 7-vec  (xyz + quat from FK)

obs dict format (matches what xrocs returns and what _update_currpos consumes):
  {
    "arm_pose": {"single": np.array(7)},          # [x, y, z, qx, qy, qz, qw]
    "arm_joints": {"single": np.array(N+1)},      # 5 calibrated joints + 1 gripper percent
    "hand_joints": {"single": np.array(1)},       # gripper percent (replicated)
    "images": {camera_key: np.ndarray(H, W, 3)},  # RGB uint8 (lerobot OpenCV camera default)
  }

Hardware: lerobot.SO101Follower (Feetech sts3215 bus over USB).
"""

import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class SO101Station:
    """xrocs-station-compatible adapter backed by lerobot.SO101Follower."""

    # 5 arm joints in LeRobot calibrated units + 1 gripper (RANGE_0_100 stroke percent)
    GRIPPER_OPEN_PCT = 30.0
    GRIPPER_CLOSED_PCT = 0.0

    def __init__(self, cfg):
        # cfg is the OmegaConf node corresponding to robot_config in so101.yaml
        self.cfg = cfg
        self.joint_dim = int(cfg.joint_dim)            # 5
        self.gripper_dim = int(cfg.gripper_dim)        # 1
        self._connected = False
        self.follower = None
        self._motor_names = None
        self.kinematics = None
        self._target_joint_degrees = None
        self._target_ee_pose = None

        self._ee_delta_scale = np.asarray(
            list(getattr(cfg, "so101_ee_delta_scale_m", [0.035, 0.030, 0.060])),
            dtype=np.float32,
        )
        assert self._ee_delta_scale.shape == (3,), (
            f"so101_ee_delta_scale_m must be length 3, got {self._ee_delta_scale.shape}"
        )
        self._ee_bounds_min = np.asarray(
            list(getattr(cfg, "so101_ee_bounds_min", getattr(cfg, "abs_pose_limit_low", [-1, -1, -1])))[
                :3
            ],
            dtype=np.float32,
        )
        self._ee_bounds_max = np.asarray(
            list(getattr(cfg, "so101_ee_bounds_max", getattr(cfg, "abs_pose_limit_high", [1, 1, 1])))[
                :3
            ],
            dtype=np.float32,
        )
        self._ee_max_delta_m = float(getattr(cfg, "so101_ee_max_delta_m", 0.08))
        self._ik_iters = int(getattr(cfg, "so101_ik_iters", 5))
        self._ik_orientation_weight = float(getattr(cfg, "so101_ik_orientation_weight", 0.0))

        # Policy action unnormalize bounds. SAC actor with use_tanh_squash=true outputs
        # joint targets in [-1, 1]. base_env._send_joint_command forwards the action
        # to station.step() unchanged, so we must map [-1, 1] -> LeRobot calibrated units here.
        # Bounds come from cube_103ep dataset min/max with a small safety buffer.
        # If cfg fields missing, use identity mapping (debug only — will not produce
        # meaningful motion).
        joint_min = getattr(cfg, "so101_joint_action_min", None)
        joint_max = getattr(cfg, "so101_joint_action_max", None)
        if joint_min is None or joint_max is None:
            import logging
            logging.warning(
                "[SO101Station] so101_joint_action_min/max not in cfg — action stays in [-1,1] "
                "range as raw LeRobot calibrated units (not physically meaningful). Set bounds in robot_type yaml."
            )
            self._unnormalize_enabled = False
        else:
            self._unnormalize_enabled = True
            self._joint_min = np.asarray(list(joint_min), dtype=np.float32)
            self._joint_max = np.asarray(list(joint_max), dtype=np.float32)
            assert self._joint_min.shape == (self.joint_dim,), (
                f"so101_joint_action_min must be length {self.joint_dim}, got {self._joint_min.shape}"
            )
            assert self._joint_max.shape == (self.joint_dim,), (
                f"so101_joint_action_max must be length {self.joint_dim}, got {self._joint_max.shape}"
            )
            self._joint_mid = (self._joint_min + self._joint_max) / 2.0
            self._joint_half = (self._joint_max - self._joint_min) / 2.0

    def _resolve_repo_path(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parent.parent / path

    def _build_kinematics(self):
        from lerobot.model.kinematics import RobotKinematics

        urdf_path = getattr(self.cfg, "so101_urdf_path", "assets/so101/so101_new_calib.urdf")
        urdf_path = self._resolve_repo_path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(
                f"SO101 URDF not found at {urdf_path}. Download so101_new_calib.urdf and mesh assets first."
            )
        return RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name=str(getattr(self.cfg, "so101_target_frame_name", "gripper_frame_link")),
            joint_names=list(self._MOTOR_NAMES_FOLLOWER[: self.joint_dim]),
        )

    # Motor order must match SO101Follower.__init__ in lerobot/robots/so101_follower/so101_follower.py.
    # Used to convert cfg.max_relative_target (list of 6) into the dict form
    # ensure_safe_goal_position expects.
    _MOTOR_NAMES_FOLLOWER = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )

    def _build_follower(self):
        # Deferred imports so the module can be imported without lerobot fully ready
        # (e.g. during fake_env dry-runs or static analysis).
        from lerobot.robots.so101_follower import SO101Follower
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.configs import Cv2Rotation

        cameras_cfg = self.cfg.so101_cameras
        camera_objs = {}
        for cam_name, cam_cfg in cameras_cfg.items():
            kw = dict(
                index_or_path=cam_cfg.index_or_path,
                width=int(cam_cfg.width),
                height=int(cam_cfg.height),
                fps=int(cam_cfg.fps),
            )
            # 透传 yaml 的 rotation（之前漏传 → fixed 一直没转正，在线吃的是横图）。
            # fork 的 OpenCVCameraConfig 无 fourcc 字段，故 fourcc 不透传（由后端默认协商）。
            rot = getattr(cam_cfg, "rotation", None)
            if rot:
                kw["rotation"] = Cv2Rotation[str(rot)]
            camera_objs[cam_name] = OpenCVCameraConfig(**kw)

        # Per-joint single-step relative-target clamp. lerobot's ensure_safe_goal_position
        # accepts float (one cap for all motors) or dict[motor_name -> cap]. cfg gives a list,
        # so convert to dict here. Without this the follower runs at native servo speed (≈300°/s),
        # which is unsafe for first-time hardware bring-up.
        max_rel = getattr(self.cfg, "max_relative_target", None)
        max_rel_dict = None
        if max_rel is not None:
            max_rel_list = [float(x) for x in list(max_rel)]
            assert len(max_rel_list) == len(self._MOTOR_NAMES_FOLLOWER), (
                f"max_relative_target length {len(max_rel_list)} != "
                f"motor count {len(self._MOTOR_NAMES_FOLLOWER)}"
            )
            max_rel_dict = dict(zip(self._MOTOR_NAMES_FOLLOWER, max_rel_list))
            logging.info(f"[SO101Station] max_relative_target (per tick): {max_rel_dict}")

        follower_cfg = SO101FollowerConfig(
            port=str(self.cfg.so101_follower_port),
            id=str(self.cfg.so101_follower_id),
            cameras=camera_objs,
            max_relative_target=max_rel_dict,
        )
        return SO101Follower(follower_cfg)

    def connect(self):
        if self._connected:
            return
        self.follower = self._build_follower()
        self.follower.connect()
        # motor order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        self._motor_names = list(self.follower.bus.motors)
        assert len(self._motor_names) == self.joint_dim + self.gripper_dim, (
            f"SO101 motor count {len(self._motor_names)} != joint_dim+gripper_dim "
            f"{self.joint_dim + self.gripper_dim}"
        )
        self.kinematics = self._build_kinematics()
        self._connected = True
        logging.info(f"[SO101Station] connected, motors: {self._motor_names}")

    def _motor_resolution(self, motor_name: str) -> float:
        motor = self.follower.bus.motors[motor_name]
        return float(self.follower.bus.model_resolution_table[motor.model] - 1)

    def _range_to_degrees(self, arm_joints: np.ndarray) -> np.ndarray:
        """Convert LeRobot RANGE_M100_100 arm joints to degrees for FK/IK.

        RobotKinematics expects degrees. SO101Follower defaults to RANGE_M100_100
        to stay compatible with native teleoperation/datasets, so kinematics must
        do this unit conversion explicitly.
        """
        arm_joints = np.asarray(arm_joints, dtype=np.float64).reshape(-1)
        out = np.zeros_like(arm_joints, dtype=np.float64)
        for i, name in enumerate(self._motor_names[: self.joint_dim]):
            cal = self.follower.calibration[name]
            val = float(np.clip(arm_joints[i], -100.0, 100.0))
            if self.follower.bus.apply_drive_mode and cal.drive_mode:
                val = -val
            raw = ((val + 100.0) / 200.0) * (cal.range_max - cal.range_min) + cal.range_min
            mid = (cal.range_min + cal.range_max) / 2.0
            out[i] = (raw - mid) * 360.0 / self._motor_resolution(name)
        return out

    def _degrees_to_range(self, arm_degrees: np.ndarray) -> np.ndarray:
        """Convert degree arm joints from IK back to LeRobot RANGE_M100_100."""
        arm_degrees = np.asarray(arm_degrees, dtype=np.float64).reshape(-1)
        out = np.zeros_like(arm_degrees, dtype=np.float64)
        for i, name in enumerate(self._motor_names[: self.joint_dim]):
            cal = self.follower.calibration[name]
            mid = (cal.range_min + cal.range_max) / 2.0
            raw = float(arm_degrees[i]) * self._motor_resolution(name) / 360.0 + mid
            val = (((raw - cal.range_min) / (cal.range_max - cal.range_min)) * 200.0) - 100.0
            if self.follower.bus.apply_drive_mode and cal.drive_mode:
                val = -val
            out[i] = np.clip(val, -100.0, 100.0)
        return out

    def _joints_for_kinematics(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64).copy()
        joints[: self.joint_dim] = self._range_to_degrees(joints[: self.joint_dim])
        return joints

    def _reset_ee_target_state(self, current: np.ndarray | None = None) -> None:
        if current is None:
            current = self._read_motor_positions()
        current_for_kinematics = self._joints_for_kinematics(current)
        self._target_joint_degrees = current_for_kinematics.copy()
        self._target_ee_pose = self.kinematics.forward_kinematics(current_for_kinematics).copy()

    def disconnect(self):
        if self.follower is not None and self._connected:
            self.follower.disconnect()
        self._connected = False

    def _extract_command(self, robot_target: dict):
        """Parse robot_target dict (xrocs-style) into (arm_cmd[5], gripper_pct[1]).

        The arm command comes in as the policy's raw output. With SAC + use_tanh_squash=true
        the actor produces values in [-1, 1]; we linearly map to LeRobot calibrated joint units
        using cube_103ep dataset min/max from so101_joint_action_min/max in cfg.

        Intervention overrides (SO101LeaderIntervention.step) already provide raw leader
        values in the same calibrated coordinate system as SO101Follower.send_action().
        """
        action_units = robot_target.get("so101_action_units")
        gripper_units = robot_target.get("so101_gripper_units", action_units)
        if "arm_joints" in robot_target and "hand_joints" in robot_target:
            arm = np.asarray(robot_target["arm_joints"]["single"]).flatten()
            gripper_raw = np.asarray(robot_target["hand_joints"]["single"]).flatten()[0]
        elif "arm" in robot_target and "position" in robot_target["arm"]:
            arm_full = np.asarray(robot_target["arm"]["position"]["single"]).flatten()
            arm = arm_full[: self.joint_dim]
            gripper_raw = arm_full[self.joint_dim]
        else:
            raise ValueError(
                f"Unsupported robot_target structure: keys={list(robot_target.keys())}"
            )

        if arm.shape[0] != self.joint_dim:
            raise ValueError(
                f"SO101 arm command dim {arm.shape[0]} != joint_dim {self.joint_dim}"
            )

        # Unnormalize: policy [-1, 1] -> LeRobot motor-normalized joint range.
        # Leader takeover and reset paths already use LeRobot motor-normalized
        # targets, the same convention as lerobot-teleoperate.
        if self._unnormalize_enabled:
            if action_units == "policy":
                arm_clipped = np.clip(arm.astype(np.float32), -1.0, 1.0)
                arm = self._joint_mid + self._joint_half * arm_clipped
            elif action_units is None:
                # Backward-compatible fallback for old callers: only treat the
                # action as policy output when it clearly lives in [-1, 1].
                looks_policy_normalized = bool(np.all(np.abs(arm) <= 1.05))
                if looks_policy_normalized:
                    arm_clipped = np.clip(arm.astype(np.float32), -1.0, 1.0)
                    arm = self._joint_mid + self._joint_half * arm_clipped

        if gripper_units == "raw":
            gripper_pct = float(np.clip(float(gripper_raw), 0.0, 100.0))
        else:
            # Policy gripper is binary in HIL-RL (0=closed, 1=open).
            gripper_pct = float(
                self.GRIPPER_OPEN_PCT if float(gripper_raw) >= 0.5 else self.GRIPPER_CLOSED_PCT
            )
        return arm.astype(np.float32), gripper_pct

    def step(self, robot_target: dict):
        """Joint command. robot_target supplied by base_env._send_joint_command."""
        if not self._connected:
            raise RuntimeError("SO101Station.step called before connect()")

        arm_cmd, gripper_pct = self._extract_command(robot_target)

        action = {f"{name}.pos": float(arm_cmd[i]) for i, name in enumerate(self._motor_names[:-1])}
        action[f"{self._motor_names[-1]}.pos"] = gripper_pct

        # A raw joint command (reset, teleop takeover, or joint-mode policy) changes
        # the physical target outside EE-delta control, so the next EE command must
        # re-anchor to the measured robot pose.
        self._target_joint_degrees = None
        self._target_ee_pose = None
        self.follower.send_action(action)
        return self.get_obs()

    def step_ee(self, robot_target: dict):
        """End-effector delta command.

        Policy action is [dx, dy, dz, gripper], where xyz is normalized to
        [-1, 1] and scaled by cfg.so101_ee_delta_scale_m. IK/FK are delegated to
        LeRobot's RobotKinematics (placo), mirroring the SO100/SO101 EE examples.
        """
        if not self._connected:
            raise RuntimeError("SO101Station.step_ee called before connect()")

        if "ee_delta" not in robot_target:
            raise ValueError(f"SO101 step_ee expects 'ee_delta', got keys={list(robot_target.keys())}")

        delta = np.asarray(robot_target["ee_delta"]["single"], dtype=np.float32).reshape(-1)
        if delta.shape[0] != 3:
            raise ValueError(f"SO101 ee_delta must be length 3, got {delta.shape}")

        units = robot_target.get("so101_ee_delta_units", "policy")
        if units == "meters":
            delta_m = delta.astype(np.float32)
        else:
            delta_m = np.clip(delta, -1.0, 1.0) * self._ee_delta_scale

        delta_norm = float(np.linalg.norm(delta_m))
        if self._ee_max_delta_m > 0 and delta_norm > self._ee_max_delta_m:
            delta_m = delta_m * (self._ee_max_delta_m / max(delta_norm, 1e-8))

        gripper_units = robot_target.get("so101_gripper_units", "policy")
        gripper_raw = np.asarray(robot_target["hand_joints"]["single"]).flatten()[0]
        if gripper_units == "raw":
            gripper_pct = float(np.clip(float(gripper_raw), 0.0, 100.0))
        else:
            gripper_pct = float(
                self.GRIPPER_OPEN_PCT if float(gripper_raw) >= 0.5 else self.GRIPPER_CLOSED_PCT
            )

        current = self._read_motor_positions()
        if self._target_joint_degrees is None or self._target_ee_pose is None:
            self._reset_ee_target_state(current)

        target_pose = self._target_ee_pose.copy()
        target_pose[:3, 3] = np.clip(
            target_pose[:3, 3] + delta_m.astype(np.float64),
            self._ee_bounds_min,
            self._ee_bounds_max,
        )

        target_joints = self._target_joint_degrees.copy()
        for _ in range(self._ik_iters):
            target_joints = self.kinematics.inverse_kinematics(
                target_joints,
                target_pose,
                position_weight=1.0,
                orientation_weight=self._ik_orientation_weight,
            ).copy()

        arm_cmd = self._degrees_to_range(target_joints[: self.joint_dim]).astype(np.float32)
        action = {f"{name}.pos": float(arm_cmd[i]) for i, name in enumerate(self._motor_names[:-1])}
        action[f"{self._motor_names[-1]}.pos"] = gripper_pct

        sent_action = self.follower.send_action(action)

        # Align with lerobot SO100FollowerEndEffector: anchor to the COMMANDED pose
        # (current_ee_pos = desired_ee_pos), NOT FK of the IK solution. Position then
        # integrates by telescoping and the held orientation stays frozen at the reset
        # orientation, so the per-tick IK residual is not accumulated into EE drift.
        # (The old FK(IK-solution) anchor accumulated residual into ~200mm drift.)
        sent_joints = np.array(
            [float(sent_action[f"{name}.pos"]) for name in self._motor_names],
            dtype=np.float64,
        )
        self._target_joint_degrees = self._joints_for_kinematics(sent_joints)
        self._target_ee_pose = target_pose
        return self.get_obs()

    def _read_motor_positions(self) -> np.ndarray:
        raw = self.follower.bus.sync_read("Present_Position")
        # placo's Boost bindings reject numpy.float32 in set_joint(); keep all
        # FK/IK inputs as float64 and cast only when sending motor commands.
        return np.array([float(raw[name]) for name in self._motor_names], dtype=np.float64)

    def _pose_from_joints(self, joints: np.ndarray) -> np.ndarray:
        if self.kinematics is None:
            raise RuntimeError("SO101 kinematics is not initialized")
        joints_for_kinematics = self._joints_for_kinematics(joints)
        T = self.kinematics.forward_kinematics(joints_for_kinematics).copy()
        quat = Rotation.from_matrix(T[:3, :3]).as_quat(canonical=True)
        return np.hstack([T[:3, 3], quat]).astype(np.float32)

    def get_obs(self):
        if not self._connected:
            raise RuntimeError("SO101Station.get_obs called before connect()")

        raw = self.follower.get_observation()

        # joint state (5 arm motors in LeRobot calibrated units + 1 gripper in 0-100 percent)
        joints_full = np.array(
            [float(raw[f"{name}.pos"]) for name in self._motor_names], dtype=np.float32
        )
        arm_joints = joints_full[: self.joint_dim]              # shape (5,)
        gripper_val = float(joints_full[self.joint_dim])         # scalar

        # cameras — lerobot OpenCVCamera defaults to RGB uint8 (H, W, 3)
        images = {}
        for cam_name in self.cfg.so101_cameras.keys():
            if cam_name in raw:
                img = raw[cam_name]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                images[cam_name] = img

        return {
            "arm_pose": {"single": self._pose_from_joints(joints_full)},
            # full 6-DoF joint vector (arm + gripper)
            "arm_joints": {"single": joints_full},
            "hand_joints": {"single": np.array([gripper_val], dtype=np.float32)},
            "images": images,
        }

    def get_ee_pose_from_joint(self, joints):
        return self._pose_from_joints(np.asarray(joints, dtype=np.float32))
