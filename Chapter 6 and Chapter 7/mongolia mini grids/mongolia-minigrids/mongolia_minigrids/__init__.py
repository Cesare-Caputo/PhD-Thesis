from gym.envs.registration import register

register(
    id='mongolia_minigrid-v0',
    entry_point='mongolia_minigrids.envs:MinigridBasic',
)

register(
    id='mongolia_minigrid-v1',
    entry_point='mongolia_minigrids.envs:MinigridMedium',
)

register(
    id='mongolia_minigrid-v2',
    entry_point='mongolia_minigrids.envs:MinigridAdvanced',
)

register(
    id='mongolia_minigrid-v3',
    entry_point='mongolia_minigrids.envs:MinigridMediumDict',
)
register(
    id='mongolia_minigrid-v4',
    entry_point='mongolia_minigrids.envs:MinigridBasicCapexSplit'
)
register(
    id='mongolia_minigrid-v5',
    entry_point='mongolia_minigrids.envs:MinigridBasic30yr'
)

register(
    id='mongolia_minigrid-v6',
    entry_point='mongolia_minigrids.envs:MinigridBasic30yr_stoch_s1'
)


register(
    id='mongolia_minigrid-v7',
    entry_point='mongolia_minigrids.envs:MinigridBasic30yr_stoch_s1_mask'
)

register(
    id='mongolia_minigrid-v8',
    entry_point='mongolia_minigrids.envs:MinigridBasic30yr_stoch_s1_mask_repl'
)


register(
    id='mongolia_minigrid-v9',
    entry_point='mongolia_minigrids.envs:MinigridBasic30yr_stoch_s2_mask_repl'
)

register(
    id='mongolia_minigrid-v10',
    entry_point='mongolia_minigrids.envs:MinigridBasic30yr_stoch_s3_mask_repl'
)

register(
    id='mongolia_minigrid-v11',
    entry_point='mongolia_minigrids.envs:MinigridBasic30yr_stoch_s4_mask_repl'
)

register(
    id='mongolia_minigrid-v12',
    entry_point='mongolia_minigrids.envs:MinigridBasic30yr_stoch_s5_mask_repl'
)


register(
    id='mongolia_minigrid-v13',
    entry_point='mongolia_minigrids.envs:MinigridBasic30yr_stoch_s6_mask_repl'
)