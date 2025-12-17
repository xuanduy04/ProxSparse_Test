import argparse

from pathlib import Path


def ckpt_type(value):
    if value in {"all", "last"}:
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "ckpt must be an integer, 'all', or 'last'"
        )


def add_ckpt_argument(parser):
    parser.add_argument("--ckpt", type=ckpt_type, default="last",
        help="Which checkpoint to load (int, 'all', or 'last')")
    return parser


def select_checkpoints(model_dir, ckpt):
    checkpoints = sorted(
        (
            p.name for p in Path(model_dir).iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-")
        ),
        key=lambda name: int(name.split("-", 1)[1])
    )
    if not checkpoints:
        raise ValueError("No checkpoint directories found.")

    if ckpt == "all":
        selected_checkpoints = checkpoints
    elif ckpt == "last":
        selected_checkpoints = [checkpoints[-1]]
    else:
        try:
            selected_checkpoints = [
                next(
                    name for name in checkpoints
                    if int(name.split("-", 1)[1]) == ckpt
                )
            ]
        except StopIteration:
            raise ValueError(f"Checkpoint {ckpt} not found.")
    return selected_checkpoints


__all__ = [
    "add_ckpt_argument",
    "select_checkpoints",
]
