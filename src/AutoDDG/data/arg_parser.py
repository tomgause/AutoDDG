import argparse


def get_argparser():
    parser = argparse.ArgumentParser("Aerial Birds Parser", add_help=False)
    parser.add_argument(
        "--dataset", required=True, type=str, help="Path to root dataset directory."
    )
    parser.add_argument(
        "--verify",
        default=False,
        action="store_true",
        help="Verify a random image chunk.",
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true",
        help="Visualize random samples after export.",
    )

    # New arguments below
    parser.add_argument(
        "--chunk-size",
        default=512,
        type=int,
        help="Tile size for chunking images (in px).",
    )
    parser.add_argument(
        "--overlap", default=32, type=int, help="Overlap in px between adjacent tiles."
    )
    parser.add_argument(
        "--export-format",
        default="datumaro",
        type=str,
        help="Export format for the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="exported_dataset",
        type=str,
        help="Path for saving exported dataset.",
    )
    parser.add_argument(
        "--random-seed", default=42, type=int, help="Random seed for reproducibility."
    )
    return parser
