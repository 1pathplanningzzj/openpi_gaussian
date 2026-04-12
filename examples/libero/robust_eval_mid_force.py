from __future__ import annotations

import tyro

from robust_eval_utils import MidForceArgs, eval_mid_force


if __name__ == "__main__":
    eval_mid_force(tyro.cli(MidForceArgs))
