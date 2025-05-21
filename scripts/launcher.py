import glob
from pathlib import Path

from pyro_eval.dataset import EvaluationDataset
from pyro_eval.evaluation import EvaluationPipeline

if __name__ == "__main__":

    device = "cpu"

    # Usage example

    # Instanciate Dataset
    dataset_path = Path(
        "./data/datasets/wildfire_test"
    )  # Folders with two sub-folders : images and labels
    print(dataset_path.exists())
    dataset = EvaluationDataset(datapath=dataset_path)
    dataset.dump()

    # Launch Evaluation

    # Compare different models
    # model_dir = "/Users/theocayla/Documents/Dev/Pyronear/models/2025_04_30_hyp-search/hyp-search-v001"
    model_dir = Path("./data/models/")
    model_paths = [str(file) for file in model_dir.glob(f"**/*.pt")]

    print(model_paths)
    #
    for model_path in model_paths:
        config = {"model_path": model_path}
        evaluation = EvaluationPipeline(dataset=dataset, config=config, device=device)
        evaluation.run()
        evaluation.save_metrics()

    for nb_consecutive_frames in [4, 5, 6, 7, 8]:
        config = {
            "model_path": model_paths[0],
            "nb_consecutive_frames": nb_consecutive_frames,
        }
        evaluation = EvaluationPipeline(dataset=dataset, config=config, device=device)
        evaluation.run()
        evaluation.save_metrics()
