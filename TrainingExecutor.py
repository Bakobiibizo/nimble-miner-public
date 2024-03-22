import os
import torch
import json
import pickle
import torch.nn as nn
import torch.distributed as dist
from torch.optim.adamw import AdamW
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, AutoFeatureExtractor
from loguru import logger
from accelerate import Accelerator, DistributedType, FullyShardedDataParallelPlugin
from accelerate.data_loader import DataLoader
from accelerate.scheduler import AcceleratedScheduler
from accelerate.state import AcceleratorState
from accelerate.accelerator import ProjectConfiguration
from tqdm.auto import tqdm
from pydantic import BaseModel
from typing import Dict


accelerator = Accelerator(
    device_placement=True,
    split_batches=True,
    mixed_precision=None,
    gradient_accumulation_steps=1,
    cpu=not torch.cuda.is_available(),
    deepspeed_plugin=None,
    fsdp_plugin=FullyShardedDataParallelPlugin(
            backward_prefetch=None,
            mixed_precision_policy=None,
            auto_wrap_policy=None,
            cpu_offload=False,
            ignored_modules=None,
            state_dict_config=None,
            optim_state_dict_config=None,
            limit_all_gathers=True,
            use_orig_params=True,
            param_init_fn=None,
            forward_prefetch=False,
            activation_checkpointing=False
        ) if DistributedType.FSDP else None,
    megatron_lm_plugin=None,
    rng_types=["cuda"],
    log_with="tensorboard",
    project_config=ProjectConfiguration(
            project_dir="nimble-bert-v2",
            logging_dir="nimble-bert-v2",
            automatic_checkpoint_naming=True,
            total_limit=5,
            save_on_each_node=False
        ),
    gradient_accumulation_plugin=None,
    dispatch_batches=False,
    even_batches=False,
    use_seedable_sampler=False,
    step_scheduler_with_optimizer=True,
    kwargs_handlers=None,
    dynamo_backend=None,
)

training_arguments = TrainingArguments(
            output_dir="nimble-bert-v2",
            overwrite_output_dir=False,
            do_train=False,
            do_eval=False,
            do_predict=False,
            evaluation_strategy="no",
            prediction_loss_only=False,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            per_gpu_train_batch_size=None,
            per_gpu_eval_batch_size=None,
            gradient_accumulation_steps=1,
            eval_accumulation_steps=None,
            eval_delay=0,
            learning_rate=5e-05,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-08,
            max_grad_norm=1.0,
            num_train_epochs=3.0,
            max_steps=-1,
            lr_scheduler_type="linear",
            lr_scheduler_kwargs={},
            warmup_ratio=0.0,
            warmup_steps=0,
            log_level="passive",
            log_level_replica="warning",
            log_on_each_node=True,
            logging_dir="nible-bert-v2",
            logging_strategy="steps",
            logging_first_step=False,
            logging_steps=500,
            logging_nan_inf_filter=True,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=None,
            save_safetensors=True,
            save_on_each_node=False,
            save_only_model=False,
            no_cuda=False,
            use_cpu=False,
            use_mps_device=False,
            seed=42,
            data_seed=None,
            jit_mode_eval=False,
            use_ipex=False,
            bf16=False,
            fp16=False,
            fp16_opt_level="O1",
            half_precision_backend="auto",
            bf16_full_eval=False,
            fp16_full_eval=False,
            tf32=None,
            local_rank=0,
            ddp_backend=None,
            tpu_num_cores=None,
            tpu_metrics_debug=False,
            debug=[],
            dataloader_drop_last=False,
            eval_steps=None,
            dataloader_num_workers=4,
            dataloader_prefetch_factor=0,
            past_index=-1,
            run_name="nible-bert-v2",
            disable_tqdm=False,
            remove_unused_columns=True,
            label_names=None,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=None,
            ignore_data_skip=False,
            fsdp=[],
            fsdp_min_num_params=0,
            fsdp_config={
                "min_num_params": 0,
                "xla": False,
                "xla_fsdp_v2": False,
                "xla_fsdp_grad_ckpt": False
            },
            fsdp_transformer_layer_cls_to_wrap= None,
            deepspeed=None,
            label_smoothing_factor=0.0,
            optim="adamw_torch",
            optim_args=None,
            adafactor=False,
            group_by_length=False,
            length_column_name="length",
            report_to=[],
            ddp_find_unused_parameters=None,
            ddp_bucket_cap_mb=None,
            ddp_broadcast_buffers=None,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=False,
            skip_memory_metrics=True,
            use_legacy_prediction_loop=False,
            push_to_hub=False,
            resume_from_checkpoint=None,
            hub_model_id=None,
            hub_strategy="every_save",
            hub_token="<HUB_TOKEN>",
            hub_private_repo=False,
            hub_always_push=False,
            gradient_checkpointing=False,
            gradient_checkpointing_kwargs=None,
            include_inputs_for_metrics=False,
            fp16_backend="auto",
            push_to_hub_model_id=None,
            push_to_hub_organization=None,
            mp_parameters="",
            auto_find_batch_size=False,
            full_determinism=False,
            torchdynamo=None,
            ray_scope="last",
            ddp_timeout=1800,
            torch_compile=False,
            torch_compile_backend=None,
            torch_compile_mode=None,
            dispatch_batches=None,
            split_batches=None,
            include_tokens_per_second=False,
            include_num_input_tokens_seen=False,
            neftune_noise_alpha=None
        )

accelerator.state.distributed_type = DistributedType.MULTI_GPU
accelerator.state._mixed_precision = "fp16"

class ExectutorTrainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.home_directory = training_arguments.output_dir
        self.training_args = training_arguments
        configs = {}
        if os.path.exists(f"{self.home_directory}/config.json"):
            with open(f"{self.home_directory}/config.json", "r") as f:
                configs = json.load(f)
        self.config = configs


        self.device = torch.device("cuda")

        self.accelerator = accelerator
        

        self.model= self.prepare_model()
        self.optimizer = self.prepare_optimizer()
        self.dataset = self.load_dataset()


    def get_config(self):
        config_path = f"{self.model_dir}/config.json"
        backup_config ={
                "model_name": "nible-bert-v2",
                "num_labels": 2,
                "num_rows": 768,
                "dataset_name": "yelp_review_full",
                "seed": 42,
                "node_url": "https://mainnet.nimble.technology:443",
            }
        try:
            with open(config_path, "rb") as f:
                config = json.loads(f.read())
            if not config:
                config = backup_config
        except ValueError as e:
            logger.error("Error loading configs.json")
            config = backup_config
        return config 

    def prepare_model(self) -> AutoModelForSequenceClassification:
        return AutoModelForSequenceClassification.from_pretrained(self.home_directory)

    def prepare_optimizer(self) -> AdamW:
        return AdamW(self.optimizer_grouped_parameters(), lr=self.training_args.learning_rate)

    def optimizer_grouped_parameters(self):
        no_decay = ["bias, LayerNorm.weight"]
        return [
            {
                "params": [
                    param
                    for name, param in self.named_parameters()
                    if all(nd not in name for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    param
                    for name, param in self.named_parameters()
                    if any(nd in name for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]


    def load_dataset(self):
        return load_dataset(self.config["finetuning_task"])
    
    def prepare_train_split(self, dataset):
        return dataset["train"].shuffle(42).select(range(1000))

    def prepare_eval_split(self, dataset):
        return dataset["test"].shuffle(42).select(range(1000))
                
    @logger.catch()
    def execute(self):
        dataset = self.load_dataset()
        train_dataset = self.prepare_train_split(dataset)
        eval_dataset = self.prepare_eval_split(dataset)

        eval_dataset, train_dataset = self.accelerator.prepare(eval_dataset, train_dataset)

        self.train()



    def model_dump(self):
        return self.model


trainer = ExectutorTrainer()
trainer.execute()