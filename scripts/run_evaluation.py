"""
CLI script to launch the evaluation of the models.
"""

import argparse
import logging
from pathlib import Path

from pyro_eval.dataset import EvaluationDataset
from pyro_eval.evaluation import EvaluationPipeline


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-models",
        help="directory containing the YOLO models to evaluate",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dir-dataset",
        help="directory containing the ultralytics dataset to evaluate the models on",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dir-save",
        help="directory to save the results",
        type=Path,
        default=Path("./data/evaluation/results/"),
    )
    parser.add_argument(
        "--device",
        help="device to use to run the evaluation pipeline.",
        choices=["cpu", "cuda", "mps"],
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not args["dir_models"].exists() and not args["dir_models"].is_dir():
        logging.error(f"invalid --dir-models")
        return False
    if not args["dir_dataset"].exists() and not args["dir_dataset"].is_dir():
        logging.error(f"invalid --dir-dataset")
        return False

    return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logger.info(args)
        device = args["device"]
        dir_dataset = args["dir_dataset"]
        dir_models = args["dir_models"]
        dir_save = args["dir_save"]
        logger.info(
            f"Evaluation of the models located in {dir_models} on the dataset {dir_dataset} running on device {device}"
        )

        # Instanciate Dataset
        dataset = EvaluationDataset(
            datapath=dir_dataset,
            dataset_ID=dir_dataset.stem,
        )
        dataset.dump()

        # Launch Evaluation

        # Compare different models
        filepaths_models = [fp for fp in dir_models.glob(f"**/*.pt")]
        logger.info(
            f"Found {len(filepaths_models)} model in {dir_models}: {filepaths_models}"
        )

        for model_path in filepaths_models:
            config = {"model_path": str(model_path)}
            logger.info(f"Evaluating the model with config {config}")
            evaluation = EvaluationPipeline(
                dataset=dataset,
                config=config,
                device=device,
            )
            evaluation.run()
            evaluation.save_metrics(dir_save)

        for nb_consecutive_frames in [4, 5, 6, 7, 8]:
            config = {
                "model_path": str(filepaths_models[0]),
                "nb_consecutive_frames": nb_consecutive_frames,
            }
            logger.info(f"Evaluating the model with config {config}")
            evaluation = EvaluationPipeline(
                dataset=dataset,
                config=config,
                device=device,
            )
            evaluation.run()
            evaluation.save_metrics(dir_save)
