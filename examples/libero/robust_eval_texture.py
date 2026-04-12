from __future__ import annotations

import tyro

from robust_eval_utils import TextureArgs, eval_texture


if __name__ == "__main__":
    eval_texture(tyro.cli(TextureArgs))
