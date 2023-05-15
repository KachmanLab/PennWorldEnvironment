from gymnasium.envs.registration import register

register(
    id="PenWorld-v0",
    entry_point="pen_world.P3:PenWorld",
)