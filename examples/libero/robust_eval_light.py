from __future__ import annotations

import tyro

from robust_eval_utils import LightArgs, eval_light


if __name__ == "__main__":
    eval_light(tyro.cli(LightArgs))
