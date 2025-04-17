import logging
import json
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm 

from pyroengine.engine import Engine

from utils import is_image, replace_bool_values

def run_engine_session(config, session_name, image_paths, labels):
    """
    Instanciate an Engine and run predictions on a list of images.
    Returns a dataframe containing image info and the confidence predicted
    """
    # TODO : better handle default values
    pyroEngine = Engine(
        nb_consecutive_frames=config.get("nb_consecutive_frames", 4),
        conf_thresh=config.get("conf_thresh", 0.15),
        max_bbox_size=config.get("max_bbox_size", 0.4),
        model_path=config.get("model_path", None)
    )

    session_results = pd.DataFrame(columns=["session", "image", "session_label", "prediction", "conf"])

    for file, label in zip(image_paths, labels):
        im = Image.open(file)
        conf = pyroEngine.predict(im)
        session_results.loc[len(session_results)] = [
            session_name,
            file,
            label != "fp",
            conf > pyroEngine.conf_thresh,
            conf
        ]

    return session_results


def run_engine_dataset(image_folder, outpath, dConfig, save_pred=True, resume=True):
    """
    Function that processes predictions through the Engine on folders of images
    TODO : rework dConfig system, should not be necessary, adapt to HF datasets
    Args
        image_folder : path toward the input folder, expects two subfolders:
            - fp : folder containing folders of image sequences containing false positive
            - fire : folder containing folders of image sequences containing identified fires
        outpath : path where data will be saved (config file, csv result file)
        dConfig : dictionnary containing different engine configuration. Each configuration can redine an attribute among
            - nb_consecutive_frames (4 by default)
            - conf_thresh (0.15 by default)
            - max_bbox_size (0.4 by default)
            - model_path : onnx format only (None by default)
        save_pred : optional param that activates file saving in 
        resume : optional param that allows to load previous predictions instead of starting from scratch
    Saves results in a csv file
    """
    
    if save_pred:
        os.makedirs(outpath, exist_ok=True)
    # Load preview configurations
    if os.path.isfile(os.path.join(outpath, "config.json")):
        with open(os.path.join(outpath, "config.json"), 'r') as fp:
            existing_config = json.load(fp)
    else:
        existing_config = {}
    
    # Run predictions for each engine configurations
    for configId, config in dConfig.items():
        logging.info(f"Running config {configId}")
        logging.info(config)

        # Update config file content
        if save_pred:
            existing_config.update(config)
            with open(os.path.join(outpath, "config.json"), 'w') as fp:
                json.dump(existing_config, fp)

        # Csv file where detailed predictions are dumpedù
        os.makedirs(os.path.join(outpath, "results"), exist_ok=True)
        outCsv = os.path.join(outpath, f"results/results_{configId}.csv")
        
        # Previous predictions are loaded if they exist and if resume is set to True
        if os.path.isfile(outCsv) and resume:
            data = pd.read_csv(outCsv)
        else:
            data = pd.DataFrame(columns=["session", "image", "session_label", "prediction", "conf"])

        # fp folder contains false positives example, label is False (no fire), otherwise it's True
        for label in ["fp", "fire"]:
            for session in tqdm(os.listdir(os.path.join(image_folder, label))):
                if session in set(data["session"].to_list()):
                    continue
                session_folder = os.path.join(image_folder, label, session)
                image_paths = [
                    os.path.join(session_folder, file)
                    for file in os.listdir(session_folder)
                    if is_image(os.path.join(session_folder, file))
                ]
                labels = [label for image in image_paths]
                session_results = run_engine_session(config, session, image_paths, labels)
                
                # Add session results to master result file
                data = pd.concat(data, session_results)
                # Checkpoint to save predictions
                if save_pred and len(data) % 50 == 0:
                    data.to_csv(outCsv, index=False)
        if save_pred:
            data.to_csv(outCsv, index=False)


def concat_preds(datapath):
    '''
    Concatenate predictions from existing result files.
    Results are expected to be stored in files named results_i.csv where i is the config identifier.
    '''
    df_final = None

    # Retrieve existing results
    predsId = [
        file.split("_")[-1].split(".")[0] # retrieve Id such as results_i.csv gives i
        for file in os.listdir(os.path.join(datapath, "results"))
        if os.path.splitext(file)[-1] == ".csv"
    ]

    for i in predsId:
        filepath = os.path.join(datapath, f"results/results_{i}.csv")
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
        else:
            logging.info(f"File not found : {filepath}")
            continue
        
        df.rename(columns={"prediction": f"prediction_{i}"}, inplace=True)
        df.rename(columns={"conf": f"conf_{i}"}, inplace=True)
        
        if df_final is None:
            df_final = df
        else:
            df_final = pd.merge(df_final, df, on=["session", "image", "session_label"], how="outer")

    column_order = [
        "session",
        "image",
        "session_label"
    ]

    column_order += [f"prediction_{i}" for i in predsId]
    column_order += [f"conf_{i}" for i in predsId]
    df_final = df_final[column_order]
    df_final.to_csv(os.path.join(datapath, "merged_preds.csv"), index=False)


def compute_event_f1_score(tp, fp, fn):
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)

def compute_accuracy(datapath):
    '''
    Computes metrics on the predictions run, from results stored in the prediction csv.
    '''
    results = {}
    df = pd.read_csv(os.path.join(datapath, "merged_preds.csv")) 
    predictions = [column for column in df.columns if "prediction_" in column]
    nb_fire = 0
    nb_non_fire = 0
    for session in set(df["session"].to_list()):
        if df.loc[df["session"] == session, "session_label"].unique()[0] == True:
            nb_fire += 1
        else:
            nb_non_fire += 1
    with open(os.path.join(datapath, "config.json"), 'r') as fp:
        dConfigs = json.load(fp)

    for predictionId in predictions:
        results[predictionId] = {
            "details" : {},
            "missed_detections" : 0,
            "false_positives" : 0,
            "config" : dConfigs.get(predictionId.split("_")[-1], {}),
        }

        for session in set(df["session"].to_list()):
            session_data = df[df["session"] == session]
            label_value = session_data["session_label"].iloc[0]
            prediction_value = session_data[predictionId].any()

            if prediction_value != label_value:
                if label_value:
                    results[predictionId]["missed_detections"] += 1
                    results[predictionId]["details"].update({session : "Missed detection"})
                else:
                    results[predictionId]["false_positives"] += 1
                    results[predictionId]["details"].update({session : "False positive"})
            else:
                if label_value:
                    results[predictionId]["details"].update({session : "Correct : Fire detectedL."})
                else:
                    results[predictionId]["details"].update({session : "Correct : No fire detected."})

        results[predictionId]["f1"] = compute_event_f1_score(
            tp=nb_fire - results[predictionId]["missed_detections"],
            fp=results[predictionId]["false_positives"],
            fn=results[predictionId]["missed_detections"]
        )

        results[predictionId]["missed_detections"] = round(results[predictionId]["missed_detections"]/nb_fire, 2)
        results[predictionId]["false_positives"] = round(results[predictionId]["false_positives"]/nb_non_fire, 2)

    with open(os.path.join(datapath, "results_merged.json"), 'w') as fp:
        json.dump(replace_bool_values(results), fp)

if __name__ == "__main__":

    # Usage example
    image_folder = "data/manual_split_original"
    datapath = "analysis/fp_analysis/250214"
    config = {
        "1" : {
            "max_bbox_size" : 0.2,
        },
    }

    run_engine_dataset(image_folder, datapath, config, save_pred=True)
    concat_preds(datapath)
    compute_accuracy(datapath)
    
