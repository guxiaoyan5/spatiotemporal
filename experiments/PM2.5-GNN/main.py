import argparse
import os.path

import yaml
from torch import nn

from dataset.HazeDataset import Graph
from dataset.dataloader import get_dataloader
from models.pm25_GNN import PM25_GNN
from train.pm25_GNN_train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="./config.yaml")
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
log_dir = os.path.join("../../logs/pm25_GNN/knowAir")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data = get_dataloader(config, config["train"]["batch_size"])
graph = Graph(config["dataset"]["altitude_fp"], config["dataset"]["city_fp"])
model = PM25_GNN(hist_len=config["experiments"]["hist_len"], pred_len=config["experiments"]["pred_len"],
                 in_dim=config["experiments"]["in_dim"], city_num=config["dataset"]["node_num"],
                 edge_index=graph.edge_index, edge_attr=graph.edge_attr, wind_mean=data["wind_mean"],
                 wind_std=data["wind_std"])
loss_fn = nn.L1Loss()
trainer = Trainer(model, loss_fn, data, config["train"]["lr"], config["train"]["steps"],
                  config["train"]["lr_decay_ratio"], log_dir, config["train"]["n_exp"], config["train"]["save_iter"],
                  None, config["train"]["epochs"], config["train"]["patience"],
                  config["train"]["device"])
trainer.train()
trainer.test("test")
