import argparse
import torch
from torch_geometric.datasets import Actor, Planetoid, WebKB, WikipediaNetwork
from sklearn.metrics import accuracy_score

from util import standardize
from graph_tree_booster import GraphTreeBooster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("dataset", default="Cora", choices=["Cora", "Citeseer", "Texas", "Cornell", "Actor", "Chameleon"])
args = parser.parse_args()
dataset_name = args.dataset

DatasetClass = {
    "Cora": Planetoid,
    "Citeseer": Planetoid,
    "Texas": WebKB,
    "Cornell": WebKB,
    "Actor": Actor,
    "Chameleon": WikipediaNetwork
}[dataset_name]

if dataset_name == "Actor":
    dataset = DatasetClass(root="data")
else:
    dataset = DatasetClass(root="data", name=dataset_name)
data = dataset[0].to(device)

if DatasetClass == Planetoid:
    train_mask = data.train_mask
    val_mask   = data.val_mask
    test_mask  = data.test_mask
else:
    train_mask = data.train_mask[:, 0]
    val_mask   = data.val_mask[:, 0]
    test_mask  = data.test_mask[:, 0]

X_std = standardize(X=data.x, edge_index=data.edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, max_dist=7)

split_idx = {
    "train": torch.where(train_mask)[0],
    "valid": torch.where(val_mask)[0],
    "test": torch.where(test_mask)[0]
}
    
tree_optimizer_kwargs = {"lr": 0.05, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
split_optimizer_kwargs = {"lr": 0.1, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.1}

model = GraphTreeBooster(
    objective="softmax_crossentropy", aggregator="heat_kernel", aggregator_hparams={"filter_order": 7}, aggregator_parity_separation=False,
    max_leaf_nodes=10,
    tree_optimizer_kwargs=tree_optimizer_kwargs,
    tree_optimizer_epochs=20,
    split_optimizer_kwargs=split_optimizer_kwargs,
    split_optimizer_epochs=50
)

model.fit(X_std, data.y, edge_index=data.edge_index, train_idx=split_idx["train"], val_idx=split_idx["valid"], tree_optimizer_early_stopping_rounds=5,
          split_optimizer_early_stopping_rounds=5, early_stopping_rounds=10, eval_metric=["accuracy", "loss", "time"])

for split_name, idx in split_idx.items():
    y_pred = model.predict_label(test_idx=idx)
    acc = accuracy_score(data.y[idx].cpu().numpy(), y_pred.cpu().numpy())
    print(f"{split_name} Accuracy: {acc:.4f}")
