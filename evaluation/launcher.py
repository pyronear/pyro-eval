import glob

from dataset import EvaluationDataset
from evaluation import EvaluationPipeline

if __name__ == "__main__":

    # Usage example
    
    # Instanciate Dataset
    dataset_path = "/Users/theocayla/Documents/Dev/Pyronear/data/test_pipeline" # Folders with two sub-folders : images and labels
    dataset = EvaluationDataset(datapath=dataset_path)
    dataset.dump()

    # Launch Evaluation

    # Compare different models
    model_dir = "/Users/theocayla/Documents/Dev/Pyronear/models/2025_04_30_hyp-search/hyp-search-v001"
    model_paths = [file for file in glob.glob(f"{model_dir}/*") if file.endswith(".pt")]

    for model_path in model_paths:
        config = {
            "model_path" : model_path
        }
        evaluation = EvaluationPipeline(dataset=dataset, config=config, device="mps")
        evaluation.run()
        evaluation.save_metrics()

    for nb_consecutive_frames in [4, 5, 6, 7, 8]:
        config = {
            "model_path" : model_paths[0],
            "nb_consecutive_frames" : nb_consecutive_frames,
        }
        evaluation = EvaluationPipeline(dataset=dataset, config=config, device="mps")
        evaluation.run()
        evaluation.save_metrics()