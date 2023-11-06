# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:02:09 2021

@author: cesa_
"""

from gym.envs.registration import register

register(
    id='wte-v0',
    entry_point='gym_wte_full.envs:WTE_EnvFull',
)


register(
    id='wte-v1',
    entry_point='gym_wte_full.envs:WTE_EnvFull_Test',
)

register(
    id='wte-v2',
    entry_point='gym_wte_full.envs:WTE_EnvFull_50',
)

register(
    id='wte-v3',
    entry_point='gym_wte_full.envs:WTE_EnvFull_debug',
)

register(
    id='wte-v4',
    entry_point='gym_wte_full.envs:WTE_EnvFull_capmax',
)