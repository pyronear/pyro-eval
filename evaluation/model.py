import logging
import os

import onnxruntime
import torch
from huggingface_hub import hf_hub_download, HfApi, HfFolder
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image as PillowImage
from ultralytics import YOLO

class Model:
    def __init__(self, model_path, inference_params, device=None):
        self.model_path = model_path
    
        self.model = self.load_model()
        self.device = self.get_device(device)
        self.model.to(self.device)
        self.format = None
        self.inference_params = self.set_inference_params(inference_params)

    def load_model(self):
        if not self.model_path:
            raise ValueError(f"No model provided for evaluation, path needs to be specified.")

        if os.path.isfile(self.model_path):
            # Local file, .onnx format
            if os.path.splitext(self.model_path)[-1] == ".onnx":
                try:
                    return onnxruntime.InferenceSession(self.model_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to load the ONNX model from {self.model_path}: {str(e)}") from e

                logging.info(f"ONNX model loaded successfully from {self.model_path}")

            # Local file, .pt format
            if os.path.splitext(self.model_path)[-1] == ".pt":
                return YOLO(self.model_path)
        else:
            # File doesn't exist, check for a HuggingFace repo - TODO : decide HF models path would be provided
            if "huggingface.co" in self.model_path:
                token = HfFolder.get_token()
                repo_id = self.model_path.split("https://huggingface.co/")[-1]
                filename = f"{os.path.basename(repo_id)}.pt"
                if token is None:
                    raise ValueError("Error : no Hugging Face found. Please authenticate with `huggingface-cli login`.")
                try:
                    hf_hub_download(repo_id=repo_id, filename=filename)
                except HfHubHTTPError as e:
                    raise ValueError(f"Access denied to  ({repo_id}): {e}")

                # Check model existence on HuggingFace
                api = HfApi()
                # Remove the first part of the url
                
                model_info = api.dataset_info(repo_id, token=token)
                if not model_info:
                    raise ValueError(f"Error : {self.model_path} doesn't exist or is not accessible.")

                # All checks are correct, return the model
                return YOLO(self.model_path)
            # File doesn't not exist, but path is not a huggingface path
            raise ValueError(f"Model file not found: {self.model_path}")

    def load_onnx(self):
        pass

    def load_pt(self):
        pass

    def load_HF(self):
        pass

    def get_device(self, device):
        """
        Returns proper devide depending on configuration
        """
        if device is not None:
            return torch.device(device)
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def set_inference_params(self, inference_params):
        return {
            "conf" : inference_params.get("conf", 0.05),
            "iou" : inference_params.get("iou", 0),
        }

    def inference(self, image: PillowImage):
        if self.format == "onnx":
            # pred = self.ort_session.run(["output0"], {"images": np_img})[0][0]
            pass
        else:
            results = self.model.predict(
                source=image,
                conf=self.inference_params["conf"],
                iou=self.inference_params["iou"],
                imgsz=self.inference_params["imgsz"],
                device=self.device
            )[0]