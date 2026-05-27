"""SO101 hardware adapter exposing xrocs-station-compatible interface.

`rl_envs.base_env.BaseEnv` expects a `self.robot_station` object with these methods:
  - connect()
  - step(robot_target) -> obs              (joint command)
  - step_ee(robot_target) -> obs           (EE-delta command, not supported in joint mode)
  - get_obs() -> obs
  - get_ee_pose_from_joint(joint) -> 7-vec  (xyz + quat, for xtele path; SO101 uses dummy zeros)

obs dict format (matches what xrocs returns and what _update_currpos consumes):
  {
    "arm_pose": {"single": np.array(7)},          # [x, y, z, qx, qy, qz, qw] — dummy in joint mode
    "arm_joints": {"single": np.array(N+1)},      # 5 calibrated joints + 1 gripper percent
    "hand_joints": {"single": np.array(1)},       # gripper percent (replicated)
    "images": {camera_key: np.ndarray(H, W, 3)},  # RGB uint8 (lerobot OpenCV camera default)
  }

Hardware: lerobot.SO101Follower (Feetech sts3215 bus over USB).
"""

import logging
import numpy as np


class SO101Station:
    """xrocs-station-compatible adapter backed by lerobot.SO101Follower.

    Only joint mode is supported. EE-delta would require URDF + IK; intentionally
    not implemented to keep the SO101 integration minimal.
    """

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

        cameras_cfg = self.cfg.so101_cameras
        camera_objs = {}
        for cam_name, cam_cfg in cameras_cfg.items():
            camera_objs[cam_name] = OpenCVCameraConfig(
                index_or_path=cam_cfg.index_or_path,
                width=int(cam_cfg.width),
                height=int(cam_cfg.height),
                fps=int(cam_cfg.fps),
            )

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
        self._connected = True
        logging.info(f"[SO101Station] connected, motors: {self._motor_names}")

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

        self.follower.send_action(action)
        return self.get_obs()

    def step_ee(self, robot_target: dict):
        raise NotImplementedError(
            "SO101 EE-delta control not implemented. Use control_mode='joint' in task yaml. "
            "Implementing this would require an SO101 URDF + IK module (mirror of "
            "SO100FollowerEndEffector in lerobot fork)."
        )

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
            # Joint mode: no forward kinematics; return identity 7-vec so base_env._update_currpos
            # can still split it into translation+quaternion without crashing.
            "arm_pose": {"single": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)},
            # full 6-DoF joint vector (arm + gripper)
            "arm_joints": {"single": joints_full},
            "hand_joints": {"single": np.array([gripper_val], dtype=np.float32)},
            "images": images,
        }

    def get_ee_pose_from_joint(self, joints):
        """Dummy FK — SO101 joint mode does not need EE pose. Returns identity 7-vec."""
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
