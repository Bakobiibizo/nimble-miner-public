import datasets
import json
import pickle
from random import random
from transformers import TrainingArguments


TASK_ARGS = {
    "model_name": "nimble-bert-v2",
    "num_labels": 3,
    "num_rows": 1000,
    "dataset_name": "yelp_review_full",
    "seed": int(random()),
    "node_url": "https://mainnet.nimble.technology:443",
    }


training_args_path = "nimble-bert-v2/training_args.bin"

def save_training_args():
    with open("nimble-bert-v2/training_args.json", "rb") as file:
        config = json.loads(file.read())
   
    with open("nimble-bert-v2/training_args.bin", "wb") as f:
        pickle.dump(TASK_ARGS, f)

def save_training_args_json():
    with open(training_args_path, "rb") as f:
        training_args_json = f.read()

    trainingargs = TrainingArguments(training_args_path.replace(".bin", ".json")).to_json_string()

    with open("training_args.json", "w", encoding="utf-8") as f:
        f.write(trainingargs)

save_training_args()
#save_training_args_json()