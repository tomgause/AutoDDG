import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

src_path = os.path.abspath(os.path.join(os.getcwd(), "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from AutoDDG.config import Config
from AutoDDG.job import Job

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Command-line utility for managing data and jobs."
    )

    parser.add_argument(
        "config", type=Path, help="Path to the configuration JSON file."
    )

    args = parser.parse_args()

    config = Config.from_json(args.config)

    Job(config).run()


if __name__ == "__main__":
    main()
