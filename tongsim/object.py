import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from isaacgym import gymapi


def get_texture_path(texture: int) -> Optional[str]:
    if texture == 0:
        return "textures/texture_table.jpg"
    elif texture == 1:
        return "textures/background_carpet.jpg"
    elif texture == 2:
        return "textures/background_texture_metal_rust.jpg"
    elif texture == 3:
        return "textures/metal_wall_iron_fence.jpg"
    elif texture == 4:
        return "textures/particle_board_paint_aged.jpg"
    elif texture == 5:
        return "textures/pebble_stone_texture_nature.jpg"
    elif texture == 6:
        return "textures/texture_aluminum.jpg"
    elif texture == 7:
        return "textures/texture_background_wall_paint_2.jpg"
    elif texture == 8:
        return "textures/texture_background_wall_paint_3.jpg"
    elif texture == 9:
        return "textures/texture_stone_stone_texture_0.jpg"
    elif texture == 10:
        return "textures/texture_wood_brown_1033760.jpg"
    else:
        return None


@dataclass
class GymObjectInstance:
    env: object
    env_idx: int
    handle: int
    position: list
    quaternion: list


class GymObject(ABC):
    asset_file: str = None

    def __init__(
        self,
        gym,
        sim,
        asset_root: str,
    ):
        self.gym = gym
        self.sim = sim

        self.asset_root = asset_root

        self.instances: List[GymObjectInstance] = []

    @abstractmethod
    def create(self, env, env_idx: int, *args, **kwargs):
        """
        Create an object in the given environment.
        """
        pass

    @abstractmethod
    def reset(self, env_idx: int, *args, **kwargs):
        """
        Reset an object in the given environment.
        """
        pass

    @abstractmethod
    def get_state(self, env_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the state of the object in the given environment.
        Returns a dictionary with keys like 'position', 'rotation', etc.
        """
        pass

    def get_instance(self, env_idx: int) -> Optional[GymObjectInstance]:
        for instance in self.instances:
            if instance.env_idx == env_idx:
                return instance
        return None

    def set_body_segmentation(self, seg_id: int):
        for instance in self.instances:
            box_dict = self.gym.get_actor_rigid_body_dict(instance.env, instance.handle)
            for key in box_dict:
                self.gym.set_rigid_body_segmentation_id(instance.env, instance.handle, box_dict[key], seg_id)


class GymMovableObject(GymObject):
    def create(self, env, env_idx: int, object_asset, name: str, position: list, quaternion: list, rgb: list = None, texture: int = None):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*position)
        pose.r = gymapi.Quat(*quaternion)
        handle = self.gym.create_actor(env, object_asset, pose, name, env_idx, 0)

        if texture is not None:
            texture_handle = self.gym.create_texture_from_file(self.sim, os.path.join(self.asset_root, get_texture_path(texture)))
            rigid_body_dict = self.gym.get_actor_rigid_body_dict(env, handle)
            for key in rigid_body_dict:
                self.gym.set_rigid_body_texture(env, handle, rigid_body_dict[key], gymapi.MESH_VISUAL_AND_COLLISION, texture_handle)

        self.instances.append(GymObjectInstance(env, env_idx, handle, position, quaternion))

        if rgb is not None:
            self.set_color(env_idx, rgb)

    def reset(self, env_idx: int, position: Optional[list] = None, quaternion: Optional[list] = None, rgb: list = None, texture: int = None):
        instance = self.get_instance(env_idx)
        if instance is not None:
            rigid_body_states = self.gym.get_actor_rigid_body_states(instance.env, instance.handle, gymapi.STATE_POS)

            position = position or instance.position
            quaternion = quaternion or instance.quaternion

            for i in range(3):
                rigid_body_states[0]["pose"]["p"][i] = position[i]
            for i in range(3):
                rigid_body_states[0]["pose"]["r"][i] = quaternion[i]
            self.gym.set_actor_rigid_body_states(instance.env, instance.handle, rigid_body_states, gymapi.STATE_POS)

            if texture is not None:
                texture_handle = self.gym.create_texture_from_file(self.sim, os.path.join(self.asset_root, get_texture_path(texture)))
                rigid_body_dict = self.gym.get_actor_rigid_body_dict(instance.env, instance.handle)
                for key in rigid_body_dict:
                    self.gym.set_rigid_body_texture(
                        instance.env, instance.handle, rigid_body_dict[key], gymapi.MESH_VISUAL_AND_COLLISION, texture_handle
                    )

            if rgb is not None:
                self.set_color(env_idx, rgb)

    def set_color(self, env_idx: int, rgb: list):
        instance = self.get_instance(env_idx)
        if instance is not None:
            color = gymapi.Vec3(*rgb)
            self.gym.set_rigid_body_color(instance.env, instance.handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    def get_state(self, env_idx: int) -> Optional[Dict[str, np.ndarray]]:
        instance = self.get_instance(env_idx)
        if instance is not None:
            rigid_body_states = self.gym.get_actor_rigid_body_states(instance.env, instance.handle, gymapi.STATE_POS)

            pos = np.array(rigid_body_states[0]["pose"]["p"].tolist())
            quat = np.array(rigid_body_states[0]["pose"]["r"].tolist())

            return {"position": pos, "rotation": quat}
        return None


class Box(GymMovableObject):
    def __init__(self, gym, sim, asset_root: str, size: float = 0.045):
        super().__init__(gym, sim, asset_root)

        # Create box asset
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0  # kg/m^3 (weight of box: 91g)

        self.size = size
        self.box_asset = self.gym.create_box(self.sim, self.size, self.size, self.size, asset_options)

    def create(self, env, env_idx: int, position: list, quaternion: list, rgb: str = None, texture: int = None):
        super().create(env, env_idx, self.box_asset, "box", position, quaternion, rgb, texture)


class Sphere(GymObject):
    def __init__(self, gym, sim, asset_root: str, radius: float = 0.04):
        super().__init__(gym, sim, asset_root)

        self.radius = radius

        asset_options = gymapi.AssetOptions()
        # asset_options.linear_damping = 0.5
        # asset_options.max_linear_velocity = 1
        # asset_options.tendon_limit_stiffness = 0

        self.sphere_asset = self.gym.create_sphere(self.sim, radius, asset_options)

    def create(self, env, env_idx: int, position: list, quaternion: list, rgb: list = None, texture: int = None):
        super().create(env, env_idx, self.sphere_asset, "sphere", position, quaternion, rgb, texture)


class Capsule(GymMovableObject):
    def __init__(self, gym, sim, asset_root: str, radius: float = 0.02, length: float = 0.03):
        super().__init__(gym, sim, asset_root)
        self.radius = radius
        self.length = length

        asset_options = gymapi.AssetOptions()
        # asset_options.linear_damping = 0.5
        # asset_options.max_linear_velocity = 1
        # asset_options.tendon_limit_stiffness = 0

        self.capsule_asset = self.gym.create_capsule(self.sim, self.radius, self.length, asset_options)

    def create(self, env, env_idx: int, position: list, quaternion: list, rgb: list = None, texture: int = None):
        super().create(env, env_idx, self.capsule_asset, "capsule", position, quaternion, rgb, texture)


class SmallPlate(GymMovableObject):
    asset_file = "urdf/objects/plate_small.urdf"

    def __init__(self, gym, sim, asset_root: str):
        super().__init__(gym, sim, asset_root)

        asset_options = gymapi.AssetOptions()
        # asset_options.disable_gravity = True

        self.plate_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)

    def create(self, env, env_idx: int, position: list, quaternion: list, rgb: list = None, texture: int = None):
        super().create(env, env_idx, self.plate_asset, "small_plate", position, quaternion, rgb, texture)


class LargePlate(GymMovableObject):
    asset_file = "urdf/objects/large_plate.urdf"

    def __init__(self, gym, sim, asset_root: str):
        super().__init__(gym, sim, asset_root)

        asset_options = gymapi.AssetOptions()
        # asset_options.disable_gravity = True

        self.plate_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)

    def create(self, env, env_idx: int, position: list, quaternion: list, rgb: list = None, texture: int = None):
        super().create(env, env_idx, self.plate_asset, "large_plate", position, quaternion, rgb, texture)


class Banana(GymObject):
    asset_file = "urdf/objects/banana_scale.urdf"

    def __init__(self, gym, sim, asset_root: str):
        super().__init__(gym, sim, asset_root)

        asset_options = gymapi.AssetOptions()

        self.banana_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)

    def create(self, env, env_idx: int, position: list, quaternion: list, rgb: list = None, texture: int = None):
        super().create(env, env_idx, self.banana_asset, "banana", position, quaternion, rgb, texture)


class Door(GymObject):
    asset_file = "urdf/objects/door.urdf"  # TODO Create urdf

    def __init__(self, gym, sim, asset_root: str):
        super().__init__(gym, sim, asset_root)

        # Load door asset
        self.asset_options = gymapi.AssetOptions()
        self.asset_options.armature = 0.01
        self.asset_options.fix_base_link = True
        self.asset_options.disable_gravity = True

        self.door_asset = self.gym.load_asset(self.sim, asset_root, self.asset_file, self.asset_options)

        # Set the door's dof properties
        self.default_dof_state = [0.0, 0.0]  # [hinge position, hinge velocity]

    def create(self, env, env_idx: int, position: list, quaternion: list):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*position)
        pose.r = gymapi.Quat(*quaternion)
        handle = self.gym.create_actor(env, self.door_asset, pose, "door", env_idx, 0)

        # color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        color = gymapi.Vec3(1, 0, 0)
        self.gym.set_rigid_body_color(env, handle, 4, gymapi.MESH_VISUAL_AND_COLLISION, color)
        self.gym.set_rigid_body_color(env, handle, 5, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))
        self.gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

        self.instances.append(GymObjectInstance(env, env_idx, handle, position, quaternion))

    def reset(self, env_idx: int, position: Optional[list] = None, quaternion: Optional[list] = None, init_dof_state: Optional[list] = None):
        door = self.get_instance(env_idx)

        if door is not None:
            # Reset the door state
            rigid_body_states = self.gym.get_actor_rigid_body_states(door.env, door.handle, gymapi.STATE_POS)

            if position is None:
                position = [door.pose.p.x, door.pose.p.y, door.pose.p.z]
            if quaternion is None:
                quaternion = [door.pose.r.x, door.pose.r.y, door.pose.r.z, door.pose.r.w]

            for i in range(3):
                rigid_body_states[0]["pose"]["p"][i] = position[i]
            for i in range(3):
                rigid_body_states[0]["pose"]["r"][i] = quaternion[i]
            self.gym.set_actor_rigid_body_states(door.env, door.handle, rigid_body_states, gymapi.STATE_POS)

            # Reset the door hinges
            self.gym.set_actor_dof_states(door.env, door.handle, init_dof_state or self.default_dof_state, gymapi.STATE_ALL)

    def get_state(self, env_idx: int) -> Optional[Dict[str, np.ndarray]]:
        door = self.get_instance(env_idx)
        if door is None:
            return None

        rigid_body_states = self.gym.get_actor_rigid_body_states(door.env, door.handle, gymapi.STATE_POS)
        rigid_body_names = self.gym.get_actor_rigid_body_names(door.env, door.handle)
        idx_knob = rigid_body_names.index("pullknob_lever")

        knob_pos = rigid_body_states[idx_knob]["pose"]["p"]
        knob_pos = np.array(knob_pos.tolist())

        door_states = self.gym.get_actor_dof_states(door.env, door.handle, gymapi.STATE_ALL)  # door joint state
        hinge_pos = np.array([s[0] for s in door_states])
        hinge_vel = np.array([s[1] for s in door_states])

        return {"knob_pos": knob_pos, "door_hinge_pos": hinge_pos[0], "door_hinge_vel": hinge_vel[0]}


class Table(GymObject):
    asset_file = "urdf/objects/table.urdf"

    def __init__(self, gym, sim, asset_root: str):
        super().__init__(gym, sim, asset_root)

        self.dims = [0.55, 0.9, 0.70]  # x, y, z sizes of the table

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        self.table_asset = self.gym.load_asset(self.sim, asset_root, self.asset_file, asset_options)

    def create(self, env, env_idx: int, position: list, quaternion: list, texture: int = 0):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*position)
        pose.r = gymapi.Quat(*quaternion)
        handle = self.gym.create_actor(env, self.table_asset, pose, "table", env_idx, 0)

        self.instances.append(GymObjectInstance(env, env_idx, handle, position, quaternion))

        self.set_texture(env_idx, texture)

    def reset(self, env_idx: int, position: Optional[list] = None, quaternion: Optional[list] = None):
        table = self.get_instance(env_idx)
        if table is not None:
            rigid_body_states = self.gym.get_actor_rigid_body_states(table.env, table.handle, gymapi.STATE_POS)

            if position is None:
                position = table.position
            if quaternion is None:
                quaternion = table.quaternion

            for i in range(3):
                rigid_body_states[0]["pose"]["p"][i] = position[i]
            for i in range(3):
                rigid_body_states[0]["pose"]["r"][i] = quaternion[i]
            self.gym.set_actor_rigid_body_states(table.env, table.handle, rigid_body_states, gymapi.STATE_POS)

    def set_texture(self, env_idx, texture: int):
        table = self.get_instance(env_idx)
        if table is not None:
            texture_table_handle = self.gym.create_texture_from_file(self.sim, os.path.join(self.asset_root, get_texture_path(texture)))
            table_dict = self.gym.get_actor_rigid_body_dict(table.env, table.handle)
            for key in table_dict:
                self.gym.set_rigid_body_texture(table.env, table.handle, table_dict[key], gymapi.MESH_VISUAL_AND_COLLISION, texture_table_handle)

    def get_state(self, env_idx: int) -> Optional[Dict[str, np.ndarray]]:
        table = self.get_instance(env_idx)
        if table is None:
            return None

        rigid_body_states = self.gym.get_actor_rigid_body_states(table.env, table.handle, gymapi.STATE_POS)
        rigid_body_names = self.gym.get_actor_rigid_body_names(table.env, table.handle)

        idx_table = rigid_body_names.index("table")
        pos = np.array(rigid_body_states[idx_table]["pose"]["p"].tolist())
        quat = np.array(rigid_body_states[idx_table]["pose"]["r"].tolist())

        return {"position": pos, "rotation": quat}
