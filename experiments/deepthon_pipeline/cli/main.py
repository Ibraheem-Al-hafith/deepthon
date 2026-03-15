from pathlib import Path
import argparse
from .commands import cmd_train, cmd_test_all, cmd_test
from ..utils.logging import get_logger, logger_from_config
from ..training.runner import ExperimentRunner
from ..config.loader import load_config

# logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Deepthon Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train Command
    train_p = subparsers.add_parser("train", help="Run training")
    train_p.add_argument("--config", required=True)
    train_p.add_argument("--dataset", default="all",nargs="+", help="Dataset name or 'all'")
    train_p.add_argument("--model", default="all",nargs="+", help="Model size or 'all'")
    train_p.add_argument("--resume", action="store_true")

    # Test Single
    test_p = subparsers.add_parser("test", help="Test specific checkpoint")
    test_p.add_argument("--config", required=True)
    test_p.add_argument("--dataset", required=True)
    test_p.add_argument("--model", required=True)
    test_p.add_argument("--ckpt", required=True)

    # Test All
    subparsers.add_parser("test-all", help="Test all models in experiment").add_argument("--config", required=True)

    args = parser.parse_args()
    logger = logger_from_config(args.config)
    logger.info("Starting Experiment")

    if args.command == "train":
        cmd_train(args.config, args.dataset, args.model, resume=args.resume)
    elif args.command == "test":
        # Pass required name/size to build the right architecture for testing
        cmd_test(args.config, args.ckpt, args.dataset, args.model)
    elif args.command == "test-all":
        cmd_test_all(args.config)


if __name__ == "__main__":
    main()
