from tongsim import TongSimEnv, SingleBoxTongSimEnv, PileBoxTongSimEnv


if __name__ == "__main__":
    kwargs = {
        "image_width": 640,
        "image_height": 480,
        "viewer": True,
        "sync_realtime": False,
    }

    # env = TongSimEnv(**kwargs)
    # env = SingleBoxTongSimEnv(box_rgb=[1, 0, 0], **kwargs)
    env = PileBoxTongSimEnv(box1_rgb=[1, 0, 0], box2_rgb=[0, 1, 0], **kwargs)

    while True:
        env.simulate()

        if env.viewer and env.gym.query_viewer_has_closed(env.viewer.viewer):
            break
