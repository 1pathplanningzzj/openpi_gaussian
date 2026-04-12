from __future__ import annotations

import tyro

from robust_eval_utils import RobustEvalArgs, eval_libero


if __name__ == "__main__":
    eval_libero(tyro.cli(RobustEvalArgs))
