"""This module contains the code to execute the task."""
import json
import sys
import os
import numpy as np
import requests
from requests import HTTPError
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer, 
    TrainingArguments
    )
from loguru import logger
from accelerate import Accelerator, DistributedType
from typing import Union, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

load_dotenv()

logger.level("ERROR")

class Miner:
    distributed_type: DistributedType
    model: Union[AutoModelForSequenceClassification, DDP]
    tokenizer: Union[AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast]
    task_args: Dict
    accelerator: Accelerator
    device: torch.device


class NimbleMiner(Miner, nn.Module):
    def __init__(self, task_args=None):
        super().__init__()
        if task_args is None:
            task_args = {
                "model_name": "my_model/outputs/2024-03-22-01-16-17-693140/best_model/pytorch_model.bin",
                "num_labels": 2,
                "num_rows": 768,
                "dataset_name": "yelp_review_full",
                "seed": 42,
                "node_url": "https://mainnet.nimble.technology:443",
            }
        self.accelerator = Accelerator()
        self.distributed_type =  DistributedType.MULTI_GPU
        self.task_args = task_args
        self.device = self.accelerator.device
        self.setup(rank=self.accelerator.local_process_index, world_size=self.accelerator.num_processes)
        self.model = self.prepare_model()
        self.tokenizer = self.prepare_tokenizer()
        
    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def compute_metrics(self, eval_predictions):
        logits, labels = eval_predictions
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}

    def prepare_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.task_args["model_name"])
    
    def prepare_model(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.task_args["model_name"], num_labels=self.task_args["num_labels"]
        )

    @logger.catch()
    def execute(self):
        self.print_colored("Starting training...", "blue")  # Blue for start


        dataset = load_dataset(self.task_args["dataset_name"])

        logger.info(f"Dataset {self.task_args['dataset_name']} loaded")

        train_dataset = dataset["train"].shuffle(self.task_args["seed"]).select(range(self.task_args["num_rows"]))
        eval_dataset = dataset["test"].shuffle(self.task_args["seed"]).select(range(self.task_args["num_labels"]))

        logger.info("Tokenizing complete")

        training_args = TrainingArguments(
            output_dir="my_model", evaluation_strategy="epoch"
        )

        eval_dataset, train_dataset = self.accelerator.prepare(eval_dataset, train_dataset, self.device)

        trainer = Trainer(
            model=self.accelerator.prepare_model(self.model),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        logger.info("Starting training run")
        trainer.train()
        
        trainer.save_model("my_model")
        logger.info("Model has been saved to ./my_model")


    def print_colored(self, text, color):
        """Print the given text in the specified ANSI color code."""
        print(f"\033[{color}m{text}\033[0m")


    def register_particle(self, addr):
        """This function inits the particle."""

        logger.info("Registering particle")
        url = f"{self.task_args['node_url']}/register_particle"

        try:
            response = requests.post(url, timeout=10, json={"address": addr})
            if response.status_code == 200:
                task = response.json()
                return task['args']
        except HTTPError as error:
            logger.error(f"Error during http request:\n{response.content}")
            raise HTTPError(
                f"Error during http request:\n{error}\nResponse:{response.content}"
            ) from error


    def complete_task(self):
        """This function completes the task."""
        logger.info("Final step")

        url = f"{self.task_args['node_url']}/complete_task"
        json_data = json.dumps({"address": os.getenv('NIMBLE_ADDR')})
        files = {
            "file1": open("my_model/config.json", "rb"),
            "file2": open("my_model/training_args.bin", "rb"),
            "r": (None, json_data, "application/json")
        }

        logger.info("making request")

        try:
            response = requests.post(url, files=files, timeout=60)
            if response.status_code == 200:
                logger.info("Complete!")
                with open("tracker.json", "r", encoding="utf-8") as f:
                    tracker = json.loads(f.read())
                    tracker["count"] += 1
                with open("tracker.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(tracker))


                return response.json()

        except HTTPError as error:
            logger.error(f"Error during http request:\n{response.content}")
            raise HTTPError(f"Error during http request:\n{error}\nResponse:{response.content}")


    def perform(self):
        if address := os.getenv("NIMBLE_ADDR"):
            logger.info(f"Address {address} started to work.")
            while True:
                try:
                    self.register_particle(address)
                    self.execute()
                    self.complete_task()
                    logger.info(f"Address {address} completed the task.")
                except Exception as e:
                    logger.error(f"Error: {e}")
        else:
            logger.error("Address not provided.")


def run():
    miner = NimbleMiner()
    miner.perform()

if __name__ == "__main__":
    run()

