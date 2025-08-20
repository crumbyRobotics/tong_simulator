import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from isaacgym import gymapi, gymtorch
import cv2

from .object import GymObjectInstance, GymObject


@dataclass
class TongRobotInstance(GymObjectInstance):
    camera: dict
    force_sensor: dict

    cam_rigid_body_idx: int


class TongRobot(GymObject):
    asset_file = "urdf/ur5/tong.urdf"

    def __init__(self, gym, sim, asset_root: str, image_width: int, image_height: int):
        super().__init__(gym, sim, asset_root)

        self.image_width = image_width
        self.image_height = image_height

        # Load ur5 asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        # asset_options.flip_visual_attachments = True # NOTE Remove 2025-06-14
        # asset_options.use_obj = True

        self.ur5_asset = self.gym.load_asset(self.sim, asset_root, self.asset_file, asset_options)

        # Configure ur5 dofs
        self.ur5_dof_props = self.gym.get_asset_dof_properties(self.ur5_asset)
        # TODO: set limits for gripper

        # Default dof states and position targets
        self.default_dof_state = [0.8, -1.5, -1.5, -1.5, 1.5, -1.5, np.deg2rad(-30), -0.8, -1.5, 1.5, -1.5, -1.5, 1.5, np.deg2rad(30), 0, -0.9]

        # Set property of all dofs
        self.ur5_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        self.ur5_dof_props["stiffness"][:].fill(400.0)  # NOTE: can be tuned
        self.ur5_dof_props["damping"][:].fill(40.0)  # NOTE: can be tuned

        # Robot pose
        self.pose = gymapi.Transform()
        self.pose.p = gymapi.Vec3(0, 0, 0.073)  # add 73 (cast) -> erase when model with cast arrive

        # Set camera properties
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.horizontal_fov = 85.0
        self.camera_props.width = self.image_width
        self.camera_props.height = self.image_height
        self.camera_props.enable_tensors = True

        self.left_cam_name = "cam0"
        self.right_cam_name = "cam1"

    def create(self, env, env_idx: int):
        # Create force sensors
        sensor_props = {"r_fixed_sensor": (0.0), "r_move_sensor": (0.0), "l_fixed_sensor": (0.0), "l_move_sensor": (0.0)}
        for sensor_name in sensor_props.keys():
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            sensor_body_idx = self.gym.find_asset_rigid_body_index(self.ur5_asset, "{}_link".format(sensor_name))
            self.gym.create_asset_force_sensor(self.ur5_asset, sensor_body_idx, sensor_pose)

        # Create actor handle
        handle = self.gym.create_actor(env, self.ur5_asset, self.pose, "ur5", env_idx, 0)

        # Get force sensors
        assert len(sensor_props.keys()) == self.gym.get_actor_force_sensor_count(env, handle)
        force_sensor = {}
        for i, sensor_name in enumerate(sensor_props.keys()):
            force_sensor[sensor_name] = self.gym.get_actor_force_sensor(env, handle, i)

        # Create camera sensors
        camera = {}
        camera[self.left_cam_name] = self.gym.create_camera_sensor(env, self.camera_props)
        camera[self.right_cam_name] = self.gym.create_camera_sensor(env, self.camera_props)
        cam_rigid_body_idx = {
            "left_cam_link": self.gym.find_actor_rigid_body_index(env, handle, "left_cam_link", gymapi.DOMAIN_SIM),
            "left_view_link": self.gym.find_actor_rigid_body_index(env, handle, "left_view_link", gymapi.DOMAIN_SIM),
            "right_cam_link": self.gym.find_actor_rigid_body_index(env, handle, "right_cam_link", gymapi.DOMAIN_SIM),
            "right_view_link": self.gym.find_actor_rigid_body_index(env, handle, "right_view_link", gymapi.DOMAIN_SIM),
        }

        # Set dof properties
        self.gym.set_actor_dof_properties(env, handle, self.ur5_dof_props)

        # Set texture for the robot
        aluminum_handle = self.gym.create_texture_from_file(self.sim, os.path.join(self.asset_root, "textures/texture_aluminum.jpg"))
        aluminum_link_list = [
            "body_link",
            "l_fixed_link",
            "l_fixed_sensor_link",
            "l_move_link",
            "l_move_sensor_link",
            "neck_yaw_link",
            "r_fixed_link",
            "r_fixed_sensor_link",
            "r_move_link",
            "r_move_sensor_link",
        ]
        ur5_dict = self.gym.get_actor_rigid_body_dict(env, handle)
        for key in aluminum_link_list:
            self.gym.set_rigid_body_texture(env, handle, ur5_dict[key], gymapi.MESH_VISUAL_AND_COLLISION, aluminum_handle)

        # Create tong robot instance
        robot = TongRobotInstance(
            env,
            env_idx,
            handle,
            [self.pose.p.x, self.pose.p.y, self.pose.p.z],
            [self.pose.r.x, self.pose.r.y, self.pose.r.z, self.pose.r.w],
            camera,
            force_sensor,
            cam_rigid_body_idx,
        )
        self.instances.append(robot)

        self.reset(env_idx)

    def reset(self, env_idx: int, init_dof_state: Optional[list] = None):
        robot = self.get_instance(env_idx)
        if robot is not None:
            self.gym.set_actor_dof_position_targets(robot.env, robot.handle, init_dof_state or self.default_dof_state)
            self.gym.set_actor_dof_states(robot.env, robot.handle, init_dof_state or self.default_dof_state, gymapi.STATE_ALL)  # 追加 2023.05.18

    def control(self, env_idx: int, dof_state: list):
        robot = self.get_instance(env_idx)
        if robot is not None:
            self.gym.set_actor_dof_position_targets(robot.env, robot.handle, dof_state)

    def set_cam_loc(self, env_idx: int):
        robot = self.get_instance(env_idx)

        if robot is not None:
            # Get the current camera links pose
            _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
            rb_states = gymtorch.wrap_tensor(_rb_states)

            p_l = rb_states[robot.cam_rigid_body_idx["left_cam_link"], :3].cpu().numpy()
            p_l_front = rb_states[robot.cam_rigid_body_idx["left_view_link"], :3].cpu().numpy()
            p_r = rb_states[robot.cam_rigid_body_idx["right_cam_link"], :3].cpu().numpy()
            p_r_front = rb_states[robot.cam_rigid_body_idx["right_view_link"], :3].cpu().numpy()

            # Set camera locations to the current camera links
            self.gym.set_camera_location(robot.camera[self.left_cam_name], robot.env, gymapi.Vec3(*p_l), gymapi.Vec3(*p_l_front))
            self.gym.set_camera_location(robot.camera[self.right_cam_name], robot.env, gymapi.Vec3(*p_r), gymapi.Vec3(*p_r_front))

    def get_state(self, env_idx: int) -> Optional[Dict[str, np.ndarray]]:
        robot = self.get_instance(env_idx)
        if robot is None:
            return None

        states = self.gym.get_actor_dof_states(robot.env, robot.handle, gymapi.STATE_ALL)
        joint_pos = np.array([s[0] for s in states])
        joint_vel = np.array([s[1] for s in states])
        return {
            "left_pos": joint_pos[7:14],
            "right_pos": joint_pos[:7],
            "neck_pos": joint_pos[14:],
            "left_vel": joint_vel[7:14],
            "right_vel": joint_vel[:7],
            "neck_vel": joint_vel[14:],
        }

    def get_force_data(self, env_idx: int) -> Optional[Dict[str, np.ndarray]]:
        robot = self.get_instance(env_idx)
        if robot is None:
            return None

        force_data = {}
        for key in robot.force_sensor.keys():
            data = robot.force_sensor[key].get_forces()
            force_data[key] = np.array([data.force.x, data.force.y, data.force.z, data.torque.x, data.torque.y, data.torque.z])
        return force_data

    def get_cam_img(self, env_idx: int) -> Optional[Dict[str, np.ndarray]]:
        robot = self.get_instance(env_idx)
        if robot is None:
            return None

        left_img = self._get_rgb_img(robot, self.left_cam_name)
        right_img = self._get_rgb_img(robot, self.right_cam_name)
        left_seg = self._get_seg_img(robot, self.left_cam_name)
        right_seg = self._get_seg_img(robot, self.right_cam_name)
        left_depth = self._get_depth_img(robot, self.left_cam_name)
        right_depth = self._get_depth_img(robot, self.right_cam_name)
        return {
            "left_img": left_img,
            "right_img": right_img,
            "left_seg": left_seg,
            "right_seg": right_seg,
            "left_depth": left_depth,
            "right_depth": right_depth,
        }

    def _get_rgb_img(self, robot: TongRobotInstance, camera_name: str) -> np.ndarray:
        img = self.gym.get_camera_image(self.sim, robot.env, robot.camera[camera_name], gymapi.IMAGE_COLOR)
        img = img.reshape(self.image_height, self.image_width, 4)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  #  <------- output type: BGR (since 2025.06.17)

    def _get_seg_img(self, robot: TongRobotInstance, camera_name: str) -> np.ndarray:
        seg = self.gym.get_camera_image(self.sim, robot.env, robot.camera[camera_name], gymapi.IMAGE_SEGMENTATION)
        seg = seg.reshape(self.image_height, self.image_width, 1)
        return np.repeat(seg, 3, axis=2)

    def _get_depth_img(self, robot: TongRobotInstance, camera_name: str) -> np.ndarray:
        depth = self.gym.get_camera_image(self.sim, robot.env, robot.camera[camera_name], gymapi.IMAGE_DEPTH)
        return np.array(depth.reshape(*depth.shape, 1) * -1000).astype(np.uint16)  # [0mm, inf)
