from __future__ import annotations

import tyro

from robust_eval_utils import InitPoseArgs, eval_init_pose


if __name__ == "__main__":
    eval_init_pose(tyro.cli(InitPoseArgs))
