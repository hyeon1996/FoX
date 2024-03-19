from functools import partial
from .multiagentenv import MultiAgentEnv
from .grf import Academy_3_vs_1_with_Keeper, Academy_Counterattack_Hard, Academy_Corner

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "academy_3_vs_1_with_keeper": partial(env_fn, env=Academy_3_vs_1_with_Keeper),
    "academy_counterattack_hard": partial(env_fn, env=Academy_Counterattack_Hard),
    "academy_corner": partial(env_fn, env=Academy_Corner),
}


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
