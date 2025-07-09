# pyro-eval

Library to evaluate Pyronear ML models 🔥

## Context

This module aims at providing an evaluation pipeline to measure and commpare
the performance of pyronear algorithms. It is split in two parts:

- Dataset management
- Metrics computation

## Installation

### Python dependencies

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, then clone this repo and install dependencies:

```bash
git clone git@github.com:earthtoolsmaker/pyro-eval.git
uv sync
```

__Note__:

This repo the [pyro-engine](https://github.com/pyronear/pyro-engine) repo as a
dependency: Make sure to run `uv sync` to retrieve changes made on this
repo.

### Data dependencies

To get the data dependencies one can use DVC - To fully use this
repository you would need access to our DVC remote storage which is
currently reserved for Pyronear members. On request, you will be provided with
AWS credentials to access our remote storage.

Pull all the data files tracked by DVC using this command:

```sh
dvc pull
```

### Scaffolding

One can use the default `./data` folder to store datasets and models to run
evaluation on:

- __Models__: One can use the `./data/models/` folder to store models to evaluate.
- __Datasets__: One can use the `./data/datasets/` folder to store the datasets.
- __Evaluation Results__: By default, the results of the evaluation runs are
stored under `./data/evaluation/`.

Example of files under `./data/`:

```bash
$ tree -L 3
.
├── datasets
│   ├── gitkeep
│   ├── wildfire_test
│   │   ├── data.yaml
│   │   ├── images
│   │   ├── labels
│   │   └── wildfire_test.csv
│   └── wildfire_test_temporal
│       ├── data.yaml
│       ├── images
│       ├── labels
│       └── wildfire_test_temporal.csv
├── evaluation
│   ├── gitkeep
│   └── runs
│       └── run-20250522-1457-7552
└── models
    ├── artistic-alpaca_v1.1.0_fe129f2.onnx
    ├── artistic-alpaca_v1.1.0_fe129f2.pt
    └── gitkeep
```

## Usage

### run_evaluation.py

This script runs the evaluation of the models on the provided test dataset.

```bash
uv run python ./scripts/run_evaluation.py \
  --dir-models ./data/models/ \
  --dir-dataset ./data/datasets/wildfire_test/ \
  --dir-temporal-dataset ./data/datasets/wildfire_test_temporal/ \
  --dir-save ./data/evaluation/runs/ \
  --device cuda \
  --loglevel info
```

## Evaluation Pipeline Design

The evaluation pipeline is composed of two steps: data preparation and metrics
computation, respectively managed by the `EvaluationDataset` and
`EvaluationPipeline` classes.

### EvaluationDataset

The `EvaluationDataset` class helps creating a custom dataset object suited for
metric computation.

The object is instanciated from an existing image folder or a hugging face
repo. A dataset ID can be passed as input, by default the id will be computed
from the current date and a custom hash of the dataset.
When instanciating from a local folder, the following rules must be follow to
ensure a proper functioning of the class:

- The root folder must contain one subfolder named `images` and one named
`labels`
- The `images` folder must contain the images files, named with the following
convention : `*_Y-m-dTH-M-S.jpg`, for example:
`seq_44_sdis-07_brison-200_2024-02-16T16-38-22.jpg`
- `labels` folder must contain a label `.txt` file in the YOLOv8 TXT format for
each image with the coordinates of the groundtruth bounding box

```txt
dataset
├── images
│   ├── image1.jpg
│   └── image2.jpg
│   └── image2.jpg
├── labels
│   ├── image1.txt
│   └── image2.txt
    └── image2.txt
```

```python
datapath = "path/to/dataset"
dataset_ID = "dataset_v0"
dataset = EvaluationDataset(datapath, dataset_ID=dataset_ID)
```

`dir-dataset` is used to evaluate the model, while `dir-temporal-dataset` is used to evaluate the engine on sequences of images.

### EvaluationPipeline

The EvaluationPipeline class helps launching the evaluation on a given dataset.
The evaluation is launched as follows:

```python
evaluation = EvaluationPipeline(dataset=dataset)
evaluation.run()
evaluation.save_metrics()
```

The complete evaluation is composed of two part : `ModelEvaluator`, which
provides metrics on the model performance alone, and `EngineEvaluator` which
provides metrics on the whole detection pipeline in the PyroEngine.

The object can be instanciated with the following parameters as input:

- `self.dataset` : `EvaluationDataset` object
- `self.config` : config dictionary as described below
- `self.run_id` : ID of the run, will be generated if not specified
- `self.use_existing_predictions` : if True, we check for existing model predicitons in the predicition folder, each prediction is saved in a json file named after the model hash. Model hash is also saved in a hashfile next to the weight file.

`config` is a dictionnary that describes the run configuration, if not in the
dictionnary, the parameters will take the default values from the Engine and Classifier classes in pyro-engine.

```json
{
    "model_path" : "path/to/model.pt",
    "model" : {
        "iou" : 0,
        "conf" : 0.15,
        "imgsz" : 1024,
    },
    "engine" : {
        "conf_thresh" : 0.15,
        "max_bbox_size" : 0.4,
        "nb_consecutive_frames" : 8,
    },
    "eval" : ["model", "engine"]
}
```

With the following keys:

- __nb_consecutive_frames__ (int): Number of consecutive frames taken into accoun in the Engine
- __conf_thresh__ (float in [0.,1.]): Confidence threshold used in the Engine, below which detections are filtered out
- __conf__ (float in [0.,1.]): Confidence threshold used in the Classifier, below which detections are filtered out
- __max_bbox_size__ (float in [0., 1.]): Bbox size above which detections are filtered out
- __iou__ (float in [0., 1.]): IoU threshold to compute matches between detected bboxes
- __eval__ (array of strs): Parts of the evaluation pipeline

### Launcher configuration

The evaluation can be launched on several configuration at once. `launcher.py`
is used to configure the runs:

```python
configs = [
        {
            "model_path" : "path/to/model_1.pt",
            "engine": {
                "conf_thresh" : 0.1,
            },
        },
        {
            "model_path" : "path/to/model_2.onnx",
            "engine": {
                "max_bbox_size" : 0.12,
            },
            "eval" : ["engine"],
        },
        {
            "model_path" : "path/to/model_3.pt",
            "model" : {
                "iou" : 0,
            },
            "eval" : ["engine"],
        },
    ]

    for config in configs:
        evaluation = EvaluationPipeline(dataset=dataset, config=config, device="mps")
        evaluation.run()
        evaluation.save_metrics()
```

### Results

Metrics are saved in the `results` folder, in a subdirectory named as the
run_ID. The data is stored in a json file with the following content.

The file contains:

- __model_metrics__ : result of ModelEvaluator
- __engine_metrics__ : result of EngineEvaluator
- __config__ : run configuration
- __dataset__ : dataset information

## Useful definitions

### EvaluationDataset()

`dataset = EvaluationDataset(datapath)`:
- `dataset.sequences`: list of image Sequence within the dataset. 
- `dataset.hash`: hash of the dataset
- `dataset.dataframe`: pandas DataFrame describing the dataset

### Sequence()

`Sequence` : object that represents a sequence of images.
- `sequence.images`: list of CustomImage objects, corresponding to image belonging to a single sequence
- `sequence.id`: name of the sequence (name of the first image without extension)
- `sequence.sequence_start`: timestamp of the first image of the sequence

### CustomImage()

`CustomImage`: object describing an image
- `image.path`: file path
- `image.sequence_id`: name of the sequence the image belongs to
- `image.timedelta`: time elapsed between the start of the sequence and this image
- `image.boxes`: ground truth coordinates
- `image.prediction` : placeholder to store a prediction
- `image.timestamp`: capture date of the image
- `image.hash`: image hash
- `image.label`: boolean label, True if wildfire present False otherwise
- `image.name`: image name 
