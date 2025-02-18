# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py

import os
import json
import time
import datetime
from pathlib import Path
import logging
import wandb
import yaml
from tqdm import tqdm

import evaluate

import re
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_rank, get_world_size
from logger import MetricLogger, SmoothedValue
from utils import get_dataloader, prepare_sample
from optims import get_optimizer, LinearWarmupCosineLRScheduler
from transformers.utils.logging import set_verbosity_error
import matplotlib.pyplot as plt
import numpy as np


def clean_text(text):
    """Lowercases, removes punctuation, and strips extra spaces."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


class Runner:
    def __init__(self, cfg, model, datasets, job_id):
        self.config = cfg

        # log
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)
        self.eval_split = self.config.config.run.eval_split
        
        # Save config to a YAML file so it can be used later to load checkpoints.
        # config_path = self.output_dir / "config.yaml"
        # with open(config_path, "w") as f:
        #     yaml.dump(self.config.to_dict(), f, default_flow_style=False)

        # settings
        self.device = torch.device(self.config.config.run.device)
        self.use_distributed = self.config.config.run.use_distributed
        self.start_epoch = 0
        self.max_epoch = self.config.config.run.optims.max_epoch
        self.evaluate_only = self.config.config.run.evaluate
        self.cuda_enabled = (self.device.type == "cuda")
        
        if self.evaluate_only:
            self.eval_dir = Path(self.config.config.run.eval_dir)
            self.eval_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.eval_dir = self.output_dir

        # test prompt
        self.prompt_template = self.config.config.model.get("prompt_template", "")
        test_prompt_path = self.config.config.model.get("test_prompt_path", "")
        if test_prompt_path:
            try:
                with open(test_prompt_path, "r") as f:
                    self.test_prompt_dict = json.load(f)
            except:
                print("Failed to load test prompt! Try to use utf-8 encoding.")
                with open(test_prompt_path, "r", encoding="utf-8") as f:
                    self.test_prompt_dict = json.load(f)
            for k in self.test_prompt_dict.keys():
                self.test_prompt_dict[k] = self.prompt_template.format(self.test_prompt_dict[k])

        else:
            self.test_prompt_dict = None

        # model
        self._model = model
        self._model.to(self.device)
        if self.use_distributed:
            self.model = DDP(
                self._model, device_ids=[self.config.config.run.gpu]
            )
        else:
            self.model = self._model

        # dataloaders
        self.train_loader = get_dataloader(datasets["train"], self.config.config.run, is_train=True, use_distributed=self.use_distributed)

        if isinstance(datasets["valid"], dict):
            self.valid_loaders = {
                name: get_dataloader(ds, self.config.config.run, is_train=False, use_distributed=self.use_distributed)
                for name, ds in datasets["valid"].items()
            }
        else:
            self.valid_loader = get_dataloader(datasets["valid"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)
            
        
        if isinstance(datasets["test"], dict):
            self.test_loaders = {
                name: get_dataloader(ds, self.config.config.run, is_train=False, use_distributed=self.use_distributed)
                for name, ds in datasets["test"].items()
            }
        else:
            self.test_loader = get_dataloader(datasets["test"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)


        # scaler
        self.use_amp = self.config.config.run.get("amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # optimizer & scheduler
        self.iters_per_epoch = len(self.train_loader) if self.config.config.run.epoch_based else self.config.config.run.iters_per_epoch
        self.optimizer = get_optimizer(self.model, self.config.config.run.optims)
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=self.config.config.run.optims.min_lr,
            init_lr=self.config.config.run.optims.init_lr,
            warmup_steps=self.config.config.run.optims.warmup_steps,
            warmup_start_lr=self.config.config.run.optims.get("warmup_start_lr", -1),
        )

        # self.log_config()
        
        if is_main_process() and not self.evaluate_only:  # Prevent multiple processes from initializing wandb
            logging.info(f"Initializing run name for wandb: {self.config.config.run.run_name}")
            wandb.init(
                project=self.config.config.run.project_name,  # Replace with your W&B project
                config=self.config.to_dict(),
                name=self.config.config.run.run_name,
                dir=str(self.output_dir),
            )

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def train_epoch(self, epoch):
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)

        for i in metric_logger.log_every(
                range(self.iters_per_epoch),
                self.config.config.run.log_freq,
                header=header,
                logger=self.log_writter,
                start_step=epoch*self.iters_per_epoch):
            if i >= self.iters_per_epoch:
                break

            samples = next(self.train_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            self.scheduler.step(cur_epoch=epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                model_output = self.model(samples)
                loss = model_output["loss"]

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.config.config.run.accum_grad_iters == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            
            # Log training metrics to wandb
            if is_main_process():
                if i % self.config.config.run.wandb_report_every == 0:
                    wandb.log({
                        "train/llm_loss": model_output.get("llm_loss", loss).item(),
                        "train/diversity_loss": model_output.get("diversity_loss", 0),
                        "train/loss": model_output.get("combined_loss", loss).item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"], 
                        "epoch": epoch, 
                        "iteration": i
                    })


        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }


    @torch.no_grad()
    def valid_epoch(self, epoch, split, dataloader=None, dataset_name="", save_json=False):
        start_time = time.time()
        model = self.unwrap_dist_model(self.model)
        model.eval()

        metric_logger = MetricLogger(delimiter="  ")
        header = f"Eval: data epoch: [{epoch}]"

        results = []
        total_samples = torch.tensor(0, dtype=torch.float32, device=self.device)
        
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
        rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge4"], use_stemmer=True)
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")

        all_references = []  # For BLEU & ROUGE
        all_hypotheses = []
        
        total_correct = torch.tensor(0, dtype=torch.float32, device=self.device)  # Exact match
        
        valid_iters = 0
        
        num_tokens = model.pool_size if model.l2p else model.num_soft_prompt_tokens
        token_use_counts = [0] * num_tokens

        for samples in tqdm(dataloader, desc=header, total=len(dataloader)):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
            prompts = [self.test_prompt_dict[task] for task in samples["task"]]

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                generated_texts, token_indices = model.generate(samples, self.config.config.run, prompts=prompts)
                if token_indices is not None:
                    for token in token_indices.view(-1).tolist():
                        token_use_counts[token] += 1
                        
                ground_truths = samples["text"]

                # Preprocess both ground truth and generated text
                generated_texts = [clean_text(text) for text in generated_texts]
                ground_truths = [clean_text(text) for text in ground_truths]

                # **Exact Match Calculation**
                exact_matches = torch.tensor(
                    [p == t for p, t in zip(generated_texts, ground_truths)],
                    dtype=torch.float32,
                    device=self.device
                )
                total_correct += exact_matches.sum()

                # **BLEU & ROUGE-L Preparation**
                all_references.extend(ground_truths)
                all_hypotheses.extend(generated_texts)

            results.append({
                "id": samples["id"],
                "ground_truth": ground_truths,
                "generated_text": generated_texts,
            })

            total_samples += len(samples["id"])
            
            # Run validation on only a subset of data to enable faster training.
            valid_iters += 1
            if self.config.config.run.num_valid_iters != -1 and valid_iters >= self.config.config.run.num_valid_iters:
                break
            
        print("Token use Counts:", token_use_counts)

        if not hasattr(self, "token_use_counts_by_dataset"):
            self.token_use_counts_by_dataset = {}

        self.token_use_counts_by_dataset[dataset_name] = token_use_counts
        
        n_datasets = len(self.valid_loaders) if hasattr(self, "valid_loaders") else 1

        # After all datasets are processed, plot the combined histogram
        if is_main_process() and len(self.token_use_counts_by_dataset) == n_datasets:
            # with open(os.path.join(self.output_dir, "token_use_counts.json"), "w") as f:
            #     json.dump(self.token_use_counts_by_dataset, f, indent=4)
            self.token_use_counts_by_dataset = {}
            
        # **Compute Corpus-Level BLEU Score**
        bleu_start_time = time.time()
        bleu_score = bleu_metric.compute(predictions=all_hypotheses, references=[[ref] for ref in all_references])["bleu"]
        bleu_time = time.time() - bleu_start_time

        # **Compute Corpus-Level ROUGE-L and Rouge-4 Score**
        rouge_start_time = time.time()
        rouge_scores = rouge_metric.compute(predictions=all_hypotheses, references=[[ref] for ref in all_references])
        total_rouge4_score = sum(
            rouge_scorer_obj.score(ref, hyp)["rouge4"].fmeasure
            for ref, hyp in zip(all_references, all_hypotheses)
        ) / len(all_references)
        rouge_time = time.time() - rouge_start_time
        
        # Compute WER 
        wer_start_time = time.time()
        wer_score = wer_metric.compute(predictions=all_hypotheses, references=all_references)
        cer_score = cer_metric.compute(predictions=all_hypotheses, references=all_references)
        wer_time = time.time() - wer_start_time

        # **Synchronize Across Distributed Processes**
        if is_dist_avail_and_initialized():
            dist.barrier()
            dist.all_reduce(total_correct)
            dist.all_reduce(torch.tensor(bleu_score, device=self.device))
            dist.all_reduce(torch.tensor(rouge_scores["rougeL"], device=self.device))
            dist.all_reduce(torch.tensor(total_rouge4_score, device=self.device))
            dist.all_reduce(torch.tensor(wer_score, device=self.device))
            dist.all_reduce(total_samples)

        # **Compute Final Scores**
        mean_exact = (total_correct / total_samples).item() if total_samples > 0 else 0.0
        mean_bleu = bleu_score if total_samples > 0 else 0.0
        mean_rouge = rouge_scores["rougeL"] if total_samples > 0 else 0.0
        mean_rouge4 = total_rouge4_score if total_samples > 0 else 0.0
        mean_wer = wer_score if total_samples > 0 else 0.0
        mean_cer = cer_score if total_samples > 0 else 0.0


        total_validation_time = time.time() - start_time
        logging.info(f"\n Dataset Name: {dataset_name}")
        logging.info(f"\n[Validation Completed] Epoch {epoch}")
        logging.info(f" - BLEU computation time: {bleu_time:.2f} seconds")
        logging.info(f" - ROUGE computation time: {rouge_time:.2f} seconds")
        logging.info(f" - Total validation time: {total_validation_time:.2f} seconds")
        logging.info(f" - WER computation time: {wer_time:.2f} seconds")
        logging.info(f" - Exact Match (Strict): {mean_exact:.4f}")
        logging.info(f" - BLEU Score: {mean_bleu:.4f}")
        logging.info(f" - ROUGE-L Score: {mean_rouge:.4f}")
        logging.info(f" - ROUGE4 Score: {mean_rouge4:.4f}")
        logging.info(f" - WER Score: {mean_wer:.4f}")
        logging.info(f" - CER Score: {mean_cer:.4f}")
        
        # **Save JSON if needed**
        if save_json and is_main_process():
            self.save_result(results, self.output_dir, f"eval_{split}_epoch_{epoch}_{dataset_name}")

        header = dataset_name if dataset_name else "val"
        # **Log Results to WandB**
        if is_main_process() and not self.evaluate_only:
            wandb.log({
                f"{header}/mean_exact_match": mean_exact,
                f"{header}/mean_bleu_score": mean_bleu,
                f"{header}/mean_rougeL_score": mean_rouge,
                f"{header}/mean_rouge4_score": mean_rouge4,
                f"{header}/mean_wer_score": mean_wer,
                f"{header}/mean_cer_score": mean_cer,
                "epoch": epoch
            })

        return {
            "mean_exact_match": mean_exact,
            "mean_bleu_score": mean_bleu,
            "mean_rougeL_score": mean_rouge,
            "mean_rouge4_score": mean_rouge4,
            "mean_wer_score": mean_wer,
            "mean_cer_score": mean_cer,
            "total_validation_time": total_validation_time,
            "token_use_counts": token_use_counts,
        }


    def save_result(self, result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        try:
            json.dump(result, open(result_file, "w"), ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Error saving {result_file}. Error: {e}")
            json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.info("rank %d starts merging results." % get_rank())
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                try:
                    res = json.load(open(result_file, "r"))
                except Exception as e:
                    logging.warning(f"Error reading {result_file}. Error: {e}")
                    res = json.load(open(result_file, "r", encoding="utf-8"))
                result += res

            try:
                json.dump(result, open(final_result_file, "w"), ensure_ascii=False)
            except Exception as e:
                logging.warning(f"Error saving {final_result_file}. Error: {e}")
                json.dump(result, open(final_result_file, "w", encoding="utf-8"), ensure_ascii=False)

            print("result file saved to %s" % final_result_file)

    def train(self):
        start_time = time.time()
        # best_agg_metric = 0
        # best_epoch = 0

        set_verbosity_error()
        # if hasattr(self, 'valid_loaders'):
        #         for ds_name, loader in self.valid_loaders.items():
        #             valid_log = self.valid_epoch(
        #                 0, "valid", dataloader=loader, dataset_name=ds_name, save_json=True)
        # else:
        #     valid_log = self.valid_epoch(
        #         0, "valid", dataloader=self.valid_loader, dataset_name=self.config.config.datasets.dataset ,save_json=True)
        split_loaders = getattr(self, f"{self.eval_split}_loaders", None)
        default_loader = getattr(self, f"{self.eval_split}_loader", None)

        valid_logs_by_dataset = {}
        if split_loaders:
            for ds_name, loader in split_loaders.items():
                valid_log = self.valid_epoch(0, self.eval_split, dataloader=loader, dataset_name=ds_name, save_json=not self.evaluate_only)
                valid_logs_by_dataset[ds_name] = valid_log
        else:
            valid_log = self.valid_epoch(0, self.eval_split, dataloader=default_loader, dataset_name=self.config.config.datasets.dataset, save_json=not self.evaluate_only)
            valid_logs_by_dataset[self.config.config.datasets.dataset] = valid_log
            
        for ds_name, valid_logs in valid_logs_by_dataset.items():
            output_path = os.path.join(self.eval_dir, f"{ds_name}_0.json")
            with open(output_path, "w") as f:
                json.dump(valid_logs, f, indent=4)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break

            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")
            
            logging.info("Validating Phase")
            set_verbosity_error()
        
            valid_logs_by_dataset = {}
            if split_loaders is not None:
                for ds_name, loader in split_loaders.items():
                    valid_log = self.valid_epoch(cur_epoch+1, self.eval_split, dataloader=loader, dataset_name=ds_name, save_json=True)
                    # Prefix the metric names with the dataset name
                    valid_log = {f"{ds_name}_{k}": v for k, v in valid_log.items()}
                    valid_logs_by_dataset[ds_name] = valid_log
                    # self.log_stats(valid_log, split_name=self.eval_split)
                    # # Optionally, use one of the metrics (say from "librisqa") for checkpointing
                    # if ds_name == "librisqa":
                    #     agg_metrics = valid_log.get("libriasr_mean_bleu_score", 0)
                    #     if agg_metrics > best_agg_metric:
                    #         best_agg_metric = agg_metrics
                    #         best_epoch = cur_epoch
                    #         self.save_checkpoint(cur_epoch, is_best=True)
            else:
                valid_log = self.valid_epoch(cur_epoch+1, self.eval_split, dataloader=default_loader, dataset_name=self.config.config.datasets.dataset, save_json=True)
                valid_logs_by_dataset[self.config.config.datasets.dataset] = valid_log
                # if valid_log is not None:
                #     if is_main_process():
                #         agg_metrics = valid_log["mean_bleu_score"]
                #         if agg_metrics > best_agg_metric:
                #             best_agg_metric = agg_metrics
                #             best_epoch = cur_epoch
                #             self.save_checkpoint(cur_epoch, is_best=True)
                #         valid_log.update({"best_epoch": best_epoch})
                #         self.log_stats(valid_log, split_name=self.eval_split)
            for ds_name, valid_logs in valid_logs_by_dataset.items():
                output_path = os.path.join(self.output_dir, f"{ds_name}_{cur_epoch+1}.json")
                with open(output_path, "w") as f:
                    json.dump(valid_logs, f, indent=4)

            self.save_checkpoint(cur_epoch, is_best=False)

            if self.use_distributed:
                dist.barrier()

        # testing phase
        # if self.evaluate_only:
        #     test_log = self.valid_epoch("best", "test", save_json=True)
        #     if is_main_process():
        #         wandb.log({
        #             "test/loss": test_log.get("loss", 0), 
        #             "test/accuracy": test_log.get("agg_metrics", 0)})


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
    
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
