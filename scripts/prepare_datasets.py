import argparse

from AutoDDG.data.AerialSeabirdsProcessor import AerialSeabirdsProcessor
from AutoDDG.data.arg_parser import get_argparser


def main(args):
    processor = AerialSeabirdsProcessor(root_dataset_path=args.dataset)
    processor.process_all(
        verify=args.verify,
        visualize=args.visualize,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        export_format=args.export_format,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AerialBirds", parents=[get_argparser()])
    args = parser.parse_args()
    main(args)
