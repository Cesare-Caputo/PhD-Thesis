from gym.envs.registration import register

register(
    id='garage-v0',
    entry_point='gym_garage.envs:GarageEnv',
)

register(
    id='garage-v1',
    entry_point='gym_garage.envs:GarageEnvFull',
)

register(
    id='garage-v2',
    entry_point='gym_garage.envs:GarageEnvFullTest',
)
