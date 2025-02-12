# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import *
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def get_paths(dataset_name):
    if dataset_name == "libriasr":
        return {
            "train": "/home/toolkit/SALMONN/data/IEMOCAP/ie-train-full.json",
            "test": "/home/toolkit/SALMONN/data/IEMOCAP/ie-test-full.json",
            "valid": "/home/toolkit/SALMONN/data/LibriASR/Librispeech-test-asr.json",
            "data_root": "/mnt/dssk/data_rw/shubham/l2p/libriSQA/",
        }
    if dataset_name == "librisqa":
        return {
            "train": "/home/toolkit/SALMONN/data/LibriSQA/LibriSQA-train.json",
            "test": "/home/toolkit/SALMONN/data/LibriSQA/LibriSQA-test.json",
            "valid": "/home/toolkit/SALMONN/data/LibriSQA/LibriSQA-test.json",
            "data_root": "/mnt/dssk/data_rw/shubham/l2p/libriSQA/",
        }
    if dataset_name == "er":
        return {
            "train": "/home/toolkit/SALMONN/data/IEMOCAP/ie-train-full.json",
            "test": "/home/toolkit/SALMONN/data/IEMOCAP/ie-test-full.json",
            "valid": "/home/toolkit/SALMONN/data/IEMOCAP/ie-valid-full.json",
            "data_root": "users/rwhetten/IEMOCAP/IEMOCAP_full_release/",
        }
    if dataset_name == "clotho_audio_cap":
        return {
            "train": "/home/toolkit/SALMONN/data/CLOTHIO_AUDIOCAP/clotho_captions_development.json",
            "test": "/home/toolkit/SALMONN/data/CLOTHIO_AUDIOCAP/clotho_captions_validation.json",
            "valid": "/home/toolkit/SALMONN/data/CLOTHIO_AUDIOCAP/clotho_captions_development.json",
            "data_root": "/mnt/dssk/data_rw/shubham/l2p/clotho/",
        }    
    


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # load config
    cfg = Config(parse_args())
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # print config
    cfg.pretty_print()

    # build model
    model = load_model(model_config)
    
    dataset_paths = get_paths(data_config.dataset)

    # build datasets
    datasets = {
        "train": SALMONNDataset(dataset_paths["train"], data_config.whisper_path, dataset_paths["data_root"]),
        "valid": SALMONNDataset(dataset_paths["valid"], data_config.whisper_path, dataset_paths["data_root"]),
        "test": SALMONNDataset(dataset_paths["test"], data_config.whisper_path, dataset_paths["data_root"]),
    }

    # build runner
    runner = Runner(cfg, model, datasets, job_id)

    # train
    runner.train()


if __name__ == "__main__":
    main()