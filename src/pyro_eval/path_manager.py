from pathlib import Path

def get_prediction_path(model_hash):
    output_dir = Path("data/predictions")
    return output_dir / f"{model_hash}.json"