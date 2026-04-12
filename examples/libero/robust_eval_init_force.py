from __future__ import annotations

import tyro

from robust_eval_utils import InitForceArgs, eval_init_force


if __name__ == "__main__":
    eval_init_force(tyro.cli(InitForceArgs))
