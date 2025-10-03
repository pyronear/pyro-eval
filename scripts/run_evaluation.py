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
        "--dir-temporal-dataset",
        help="directory containing the temporal dataset to evaluate the engine on",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dir-save",
        help="directory to save the results",
        type=Path,
        default=Path("./data/evaluation/runs/"),
    )
    parser.add_argument(
        "--device",
        help="device to use to run the evaluation pipeline.",
        choices=["cpu", "cuda", "mps"],
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--nb-consecutive-frames",
        help="number of consecutive frames taken into accoun in the Engine.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--engine-conf-thresh",
        help="confidence threshold used in the Engine, below which detections are filtered out.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-bbox-size",
        help="bbox size above which detections are filtered out.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--iou",
        help="IoU threshold to compute matches between detected bboxes.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--model-conf",
        help="confidence threshold used in the Classifier, below which detections are filtered out.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--imgsz",
        help="image size used in the model.",
        type=int,
        default=None,
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
        logging.error("invalid --dir-models")
        return False
    if not args["dir_dataset"].exists() and not args["dir_dataset"].is_dir():
        logging.error("invalid --dir-dataset")
        return False
    if (
        not args["dir_temporal_dataset"].exists()
        and not args["dir_temporal_dataset"].is_dir()
    ):
        logging.error("invalid --dir-temporal-dataset")
        return False

    return True


def get_config(args: dict) -> bool:
    """
    Builds config dict from arguments
    """
    config = {
        "engine": {},
        "model": {},
    }
    if args.get("nb_consecutive_frames"):
        config["engine"]["nb_consecutive_frames"] = args.get("nb_consecutive_frames")
    if args.get("engine_conf_thresh"):
        config["engine"]["conf_thresh"] = args.get("nb_consecutive_frames")
    if args.get("max_bbox_size"):
        config["engine"]["max_bbox_size"] = args.get("max_bbox_size")
    if args.get("iou"):
        config["model"]["iou"] = args.get("iou")
    if args.get("model_conf"):
        config["model"]["conf"] = args.get("model_conf")
    if args.get("imgsz"):
        config["model"]["imgsz"] = args.get("imgsz")

    return config


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
        dir_temporal_dataset = args["dir_temporal_dataset"]
        dir_models = args["dir_models"]
        dir_save = args["dir_save"]
        logger.info(
            f"Evaluation of the models located in {dir_models} on the dataset {dir_dataset} running on device {device}"
        )

        # Instanciate Dataset
        dataset = {
            "model": EvaluationDataset(
                datapath=dir_dataset,
                dataset_ID=dir_dataset.stem,
            ),
            "engine": EvaluationDataset(
                datapath=dir_temporal_dataset,
                dataset_ID=dir_temporal_dataset.stem,
            ),
        }

        # Launch Evaluation

        # Compare different models
        filepaths_models = [fp for fp in dir_models.glob("**/*.pt")]
        logger.info(
            f"Found {len(filepaths_models)} model in {dir_models}: {filepaths_models}"
        )
        config = get_config(args)
        for model_path in filepaths_models:
            config["model_path"] = str(model_path)
            logger.info(f"Evaluating the model with config {config}")
            evaluation = EvaluationPipeline(
                dataset=dataset,
                config=config,
                device=device,
            )
            evaluation.run()
            evaluation.save_metrics(dir_save)


# poetry run python ./scripts/run_evaluation.py \
#   --dir-models ./data/models/selected \
#   --dir-dataset ./data/datasets/wildfire_test/ \
#   --dir-temporal-dataset ./data/datasets/wildfire_test_temporal/ \
#   --dir-save ./data/evaluation/runs/ \
#   --device cuda \
#   --loglevel info
