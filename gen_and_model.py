#!/usr/bin/python3

"""
gen_and_model.py:
Single-process orchestration for:
1. loading generator/model code into memory,
2. snapshotting config files at job start,
3. generating train/test datasets,
4. running model training and evaluation.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from generator import load_config_file, run_generator_from_config
from cnn_model.main import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "training_set.json"
DEFAULT_TEST_CONFIG = PROJECT_ROOT / "configs" / "testing_set.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parses the command line arguments if entered through CLI.

    Args:
        argv (Optional[Sequence[str]], optional): Command line arguments. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate training/testing datasets and run the CNN model "
            "in a single Python process."
        )
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Root directory under which train/test dataset folders will be created.",
    )
    parser.add_argument(
        "--train-config-file",
        type=str,
        default=str(DEFAULT_TRAIN_CONFIG),
        help="Path to the training dataset JSON configuration.",
    )
    parser.add_argument(
        "--test-config-file",
        type=str,
        default=str(DEFAULT_TEST_CONFIG),
        help="Path to the testing dataset JSON configuration.",
    )
    parser.add_argument(
        "--train-subdir",
        type=str,
        default="train",
        help="Subdirectory name for the generated training dataset.",
    )
    parser.add_argument(
        "--test-subdir",
        type=str,
        default="test",
        help="Subdirectory name for the generated testing dataset.",
    )
    parser.add_argument(
        "--train-rng-seed",
        type=int,
        default=2026,
        help="RNG seed for training dataset generation.",
    )
    parser.add_argument(
        "--test-rng-seed",
        type=int,
        default=2027,
        help="RNG seed for testing dataset generation.",
    )
    parser.add_argument(
        "--model-rng-seed",
        type=int,
        default=2028,
        help="Optional override for the model RNG seed.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Main function to orchestrate dataset generation and model training/evaluation.

    Args:
        argv (Optional[Sequence[str]], optional): Command line arguments. Defaults to None.
    """
    args = parse_args(argv)

    root_dir = Path(args.dir).expanduser().resolve()
    train_dir = root_dir / args.train_subdir
    test_dir = root_dir / args.test_subdir

    # Snapshot both JSON configs immediately at process start so later edits
    # to the files do not affect this running job.
    test_config_snapshot = load_config_file(args.test_config_file)
    train_config_snapshot = load_config_file(args.train_config_file)

    print("Using run directories:")
    print(f"\tTraining Data: {train_dir}")
    print(f"\tTesting Data:  {test_dir}")
    print("\n")

    # Generate test then train, matching your current shell-script order.
    test_generation_config = run_generator_from_config(
        raw_config=test_config_snapshot,
        save_dir=str(test_dir),
        rng_seed=args.test_rng_seed,
    )
    train_generation_config = run_generator_from_config(
        raw_config=train_config_snapshot,
        save_dir=str(train_dir),
        rng_seed=args.train_rng_seed,
    )

    # map_config() may rewrite save paths (e.g., timestamp suffixes) when a
    # target directory already exists. Use the resolved output paths to ensure
    # model training reads the actual generated datasets.
    resolved_train_dir = train_generation_config["savepath"]
    resolved_test_dir = test_generation_config["savepath"]

    print("Resolved dataset directories after generation:")
    print(f"\tTraining Data: {resolved_train_dir}")
    print(f"\tTesting Data:  {resolved_test_dir}")
    print("\n")

    run_pipeline(
        train_dir=resolved_train_dir,
        test_dir=resolved_test_dir,
        rng_seed=args.model_rng_seed,
    )


if __name__ == "__main__":
    main()
