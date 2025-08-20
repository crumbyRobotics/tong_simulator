import os
import numpy as np
from isaacgym import gymapi, gymutil
from typing import Dict
import time
import zmq

from .robot import TongRobot
from .object import Box, Sphere, Capsule, SmallPlate, LargePlate, Banana, Door, Table
from .communication import SocketManager
from .viewer import Viewer


####################
###   Base Env   ###
####################
class TongSimEnv:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        viewer: bool,
        sync_realtime: bool,
        sim_freq: int = 60,
        cycle_freq: int = 10,
        spacing: float = 1.0,
        ip_addr: str = "localhost:5555",
    ):
        # Launch communication
        self.socket_manager = SocketManager(ip_addr)

        # Parse isaacgym arguments
        args = gymutil.parse_arguments()
        print("Using args:", args)

        # Simulation parameters
        self.sim_freq = sim_freq  # isaacgym simulation frequency (hz)
        self.cycle_freq = cycle_freq  # control cycle frequency (hz)
        self.sync_realtime = sync_realtime

        self.image_height = image_height  # px
        self.image_width = image_width  # px

        # Acquire gym
        self.gym = gymapi.acquire_gym()

        # Configure sim
        self.sim_params = gymapi.SimParams()
        self.config_sim_params(args)

        # Create sim
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params)
        assert self.sim is not None, "Failed to create simulation"

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Device
        self.device = "cuda" if self.sim_params.use_gpu_pipeline else "cpu"

        self.set_light()

        # Create environment
        self.asset_root = os.path.join(os.path.dirname(__file__), os.path.pardir, "assets")

        self.num_envs = 1  # NOTE Single env only
        num_per_row = 1  # NOTE Single row only

        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # Create tong robot
        self.tong_robot = TongRobot(self.gym, self.sim, self.asset_root, self.image_width, self.image_height)
        self.tong_robot.create(self.env, env_idx=0)

        # Create objects
        self.create_objects()

        # Create viewer
        self.viewer = self.create_viewer() if viewer else None

        # Launch simulator
        for _ in range(5):
            self.step_sim()

            self.tong_robot.set_cam_loc(env_idx=0)
            self.render_graphics()

            if self.viewer is not None:
                self.viewer.update()

    def simulate(self):
        imgs = self.tong_robot.get_cam_img(env_idx=0)
        states = self.tong_robot.get_state(env_idx=0)
        forces = self.tong_robot.get_force_data(env_idx=0)

        # rad2deg for gripper angles
        states["left_pos"][-1] = np.rad2deg(states["left_pos"][-1])
        states["right_pos"][-1] = np.rad2deg(states["right_pos"][-1])
        states["left_vel"][-1] = np.rad2deg(states["left_vel"][-1])
        states["right_vel"][-1] = np.rad2deg(states["right_vel"][-1])

        # (posN, posL, posR, velN, velL, velR, fsL, fsR, SbSResultL, SbSResultR, SbSResultD)
        robot_data = (
            states["neck_pos"],  # (2,)
            states["left_pos"],  # (7,)
            states["right_pos"],  # (7,)
            states["neck_vel"],  # (2,)
            states["left_vel"],  # (7,)
            states["right_vel"],  # (7,)
            np.concatenate([forces["l_fixed_sensor"], forces["l_move_sensor"], [time.time()]]),  # (12*2+1,)
            np.concatenate([forces["r_fixed_sensor"], forces["r_move_sensor"], [time.time()]]),  # (12*2+1,)
            imgs["left_img"],  # (H, W, 3)
            imgs["right_img"],  # (H, W, 3)
            imgs["left_depth"],  # (H, W, 1)
        )
        # Dict of object states
        world_data = self.get_world_state()

        try:
            self.socket_manager.send(robot_data, world_data)

            _, cmdN, cmdL, cmdR, reset = self.socket_manager.receive()
        except zmq.error.Again:
            return

        # deg2rad for gripper angles
        cmdL[-1] = np.deg2rad(cmdL[-1])
        cmdR[-1] = np.deg2rad(cmdR[-1])

        command = np.concatenate([cmdR, cmdL, cmdN]).tolist()  # (16,)

        if reset:
            self.reset(command)
        else:
            self.step(command)

    def step(self, command: np.ndarray):
        # Set joint position target
        self.tong_robot.control(env_idx=0, dof_state=command)

        # Carry out physical simulation for t times
        for _ in range(self.sim_freq // self.cycle_freq):
            self.step_sim()

        self.tong_robot.set_cam_loc(env_idx=0)
        self.render_graphics()

        # Update viewer
        if self.viewer is not None:
            self.viewer.update()

    def reset(self, command: np.ndarray):
        self.tong_robot.reset(env_idx=0, init_dof_state=command)
        self.reset_objects()

        for _ in range(5):
            self.step_sim()

        self.tong_robot.set_cam_loc(env_idx=0)
        self.render_graphics()

        # Update viewer
        if self.viewer is not None:
            self.viewer.update()

    def close(self):
        if self.viewer is not None:
            self.viewer.destroy()
        self.gym.destroy_sim(self.sim)

    def create_objects(self):
        pass

    def reset_objects(self):
        pass

    def get_world_state(self) -> Dict[str, np.ndarray]:
        return {}

    def create_viewer(self):
        cam_pos = gymapi.Vec3(2, 0, 1.5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        viewer = Viewer(self.gym, self.sim, self.env, cam_pos, cam_target)
        self.gym.prepare_sim(self.sim)
        self.tong_robot.set_body_segmentation(4294967295)
        return viewer

    def config_sim_params(self, args):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / self.sim_freq
        self.sim_params.substeps = 8
        # self.sim_params.use_gpu_pipeline = False

        assert args.physics_engine == gymapi.SIM_PHYSX
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.physx.contact_offset = 0.001
        self.sim_params.physx.friction_offset_threshold = 0.001
        self.sim_params.physx.friction_correlation_distance = 0.0005
        self.sim_params.physx.num_threads = args.num_threads
        self.sim_params.physx.use_gpu = args.use_gpu

    def step_sim(self):
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # Refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def render_graphics(self):
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

    def set_light(self):
        intensity = gymapi.Vec3(0.3, 0.3, 0.3)
        ambient = gymapi.Vec3(0.5, 0.5, 0.5)
        direction = gymapi.Vec3(1.0, 1.0, 1.0)
        self.gym.set_light_parameters(self.sim, 0, intensity, ambient, direction)
        direction = gymapi.Vec3(1.0, -1.0, 1.0)
        self.gym.set_light_parameters(self.sim, 1, intensity, ambient, direction)
        direction = gymapi.Vec3(-1.0, 1.0, 1.0)
        self.gym.set_light_parameters(self.sim, 2, intensity, ambient, direction)
        direction = gymapi.Vec3(-1.0, -1.0, 1.0)
        self.gym.set_light_parameters(self.sim, 3, intensity, ambient, direction)

    def pos2pixel(self, position: list) -> np.ndarray:
        position = np.array(position)
        pixel = np.zeros(4)
        cameras = [
            self.tong_robot.instances[0].camera[self.tong_robot.left_cam_name],
            self.tong_robot.instances[0].camera[self.tong_robot.right_cam_name],
        ]
        for i in range(2):
            projection_matrix = np.matrix(self.gym.get_camera_proj_matrix(self.sim, self.env, cameras[i]))
            view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, self.env, cameras[i]))
            point_2d = np.dot(np.dot(np.concatenate((position, [1])), view_matrix), projection_matrix)
            u = point_2d[0, 0] / point_2d[0, 3]
            v = point_2d[0, 1] / point_2d[0, 3]
            pixel[i * 2 : i * 2 + 2] = 0.5 * np.array([u + 1, 1 - v])  # [0, 1), normalized pixel cood
        return pixel


###########################
###   Customized Envs   ###
###########################
class SingleBoxTongSimEnv(TongSimEnv):
    def __init__(self, box_rgb: list, *args, **kwargs):
        self.box_rgb = box_rgb

        super().__init__(*args, **kwargs)

    def create_objects(self):
        table_pos = [0.655, -0.035, 0]
        table_quat = [0, 0, 0, 1]
        self.table = Table(self.gym, self.sim, self.asset_root)
        self.table.create(
            env=self.env,
            env_idx=0,
            position=table_pos,
            quaternion=table_quat,
        )

        box_size = 0.045
        box_pos = [table_pos[0] + np.random.uniform(-0.05, 0.05), table_pos[1] + np.random.uniform(-0.1, 0.1), self.table.dims[2] + box_size / 2]
        theta = np.random.uniform(0, np.pi / 4)
        box_quat = [0, 0, np.sin(theta), np.cos(theta)]
        self.box = Box(self.gym, self.sim, self.asset_root, box_size)
        self.box.create(
            env=self.env,
            env_idx=0,
            position=box_pos,
            quaternion=box_quat,
            rgb=self.box_rgb,
        )

    def reset_objects(self):
        self.table.reset(0)

        box_pos = [0.655 + np.random.uniform(-0.05, 0.05), -0.035 + np.random.uniform(-0.1, 0.1), self.table.dims[2] + self.box.size / 2]
        box_quat = [1, 1, 1, np.random.uniform(0, 1)]
        self.box.reset(
            env_idx=0,
            position=box_pos,
            quaternion=box_quat,
        )

    def get_world_state(self) -> Dict[str, np.ndarray]:
        table_state = self.table.get_state(env_idx=0)
        box_state = self.box.get_state(env_idx=0)

        box_state.update({"pixel": self.pos2pixel(box_state["position"])})

        world_state = {
            "table": table_state,
            "box": box_state,
        }

        return world_state


class PileBoxTongSimEnv(TongSimEnv):
    def __init__(self, box1_rgb: list, box2_rgb: list, *args, **kwargs):
        self.box1_rgb = box1_rgb
        self.box2_rgb = box2_rgb

        super().__init__(*args, **kwargs)

    def create_objects(self):
        table_pos = [0.655, -0.035, 0]
        table_quat = [0, 0, 0, 1]
        self.table = Table(self.gym, self.sim, self.asset_root)
        self.table.create(
            env=self.env,
            env_idx=0,
            position=table_pos,
            quaternion=table_quat,
        )

        box1_size = 0.045
        box1_pos = [
            table_pos[0] + np.random.uniform(-0.05, 0.05),
            table_pos[1] + np.random.uniform(-0.1, 0.1),
            self.table.dims[2] + box1_size / 2,
        ]
        theta = np.random.uniform(0, np.pi / 4)
        box1_quat = [0, 0, np.sin(theta), np.cos(theta)]
        self.box1 = Box(self.gym, self.sim, self.asset_root, box1_size)
        self.box1.create(
            env=self.env,
            env_idx=0,
            position=box1_pos,
            quaternion=box1_quat,
            rgb=self.box1_rgb,
        )

        box2_size = 0.045
        box2_pos = [
            table_pos[0] + np.random.uniform(-0.05, 0.05),
            table_pos[1] - 0.2 + np.random.uniform(-0.05, 0.05),
            self.table.dims[2] + box2_size / 2,
        ]
        theta = np.random.uniform(0, np.pi / 4)
        box2_quat = [0, 0, np.sin(theta), np.cos(theta)]
        self.box2 = Box(self.gym, self.sim, self.asset_root, box2_size)
        self.box2.create(
            env=self.env,
            env_idx=0,
            position=box2_pos,
            quaternion=box2_quat,
            rgb=self.box2_rgb,
        )

    def reset_objects(self):
        self.table.reset(env_idx=0)

        box1_pos = [
            self.table.instances[0].position[0] + np.random.uniform(-0.05, 0.05),
            self.table.instances[0].position[1] + np.random.uniform(-0.1, 0.1),
            self.table.dims[2] + self.box1.size / 2,
        ]
        theta = np.random.uniform(0, np.pi / 4)
        box1_quat = [0, 0, np.sin(theta), np.cos(theta)]
        self.box1.reset(
            env_idx=0,
            position=box1_pos,
            quaternion=box1_quat,
        )

        box2_pos = [
            self.table.instances[0].position[0] + np.random.uniform(-0.05, 0.05),
            self.table.instances[0].position[1] - 0.2 + np.random.uniform(-0.05, 0.05),
            self.table.dims[2] + self.box2.size / 2,
        ]
        theta = np.random.uniform(0, np.pi / 4)
        box2_quat = [0, 0, np.sin(theta), np.cos(theta)]
        self.box2.reset(
            env_idx=0,
            position=box2_pos,
            quaternion=box2_quat,
        )

    def get_world_state(self) -> Dict[str, np.ndarray]:
        table_state = self.table.get_state(env_idx=0)
        box1_state = self.box1.get_state(env_idx=0)
        box2_state = self.box2.get_state(env_idx=0)

        box1_state.update({"pixel": self.pos2pixel(box1_state["position"])})
        box2_state.update({"pixel": self.pos2pixel(box2_state["position"])})

        world_state = {
            "table": table_state,
            "box1": box1_state,
            "box2": box2_state,
        }

        return world_state
