import argparse
import sys

import torch
from datumaro import Dataset

sys.path.append(".")
from AutoDDG.DAVE.models.dave import build_eval as build_eval
from AutoDDG.DAVE.utils.arg_parser import get_argparser
from AutoDDG.eval.DAVEFewShotEvaluator import DAVEFewShotEvaluator
from AutoDDG.utils.logging import setup_logging


@torch.no_grad()
def main(args):
    global fig, ax, ix, iy
    ix, iy = 0, 0

    logger = setup_logging(__name__)

    # Suppose we have a Datumaro dataset and a DAVE model:
    dataset = Dataset.import_from(args.datumaro_path, format=args.datumaro_format)
    device = 0  # int(args.device) if args.device in ["0", "1"] else "cpu"
    model = build_eval(device, args)

    evaluator = DAVEFewShotEvaluator(
        dataset=dataset,
        model=model,
        device=device,
        reference_image_path=args.reference_image_path,
        save_visualizations=True,
    )

    # Run evaluation
    results = evaluator.evaluate()
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DAVE", parents=[get_argparser()])
    args = parser.parse_args()
    main(args)
