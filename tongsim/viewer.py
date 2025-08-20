from isaacgym import gymapi


class Viewer:
    def __init__(self, gym, sim, env, cam_pos, cam_target):
        self.gym = gym
        self.sim = sim

        self.viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        self.gym.viewer_camera_look_at(self.viewer, env, cam_pos, cam_target)

    def update(self):
        self.gym.clear_lines(self.viewer)
        self.gym.draw_viewer(self.viewer, self.sim, False)

    def destroy(self):
        self.gym.destroy_viewer(self.viewer)
