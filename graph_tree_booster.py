from __future__ import annotations

import sys
import heapq
from collections import deque, namedtuple
from dataclasses import dataclass
import operator

INF = sys.maxsize

import time

import torch
import torch.nn.functional as F
import numpy as np
import scipy
from torch_geometric.utils import degree, to_dense_adj
from torch_sparse import SparseTensor
from sklearn.metrics import accuracy_score

class SoftmaxCrossEntropy:
    def __init__(self, y_train):
        device = y_train.device

        y_train = y_train.squeeze().to(torch.long)
        self.y_train = y_train
        self.num_train_samples_range = torch.arange(y_train.size(0), device=device, dtype=torch.long)
        self.num_classes = int(y_train.max().item()) + 1
        
        counts = torch.bincount(y_train, minlength=self.num_classes)
        probs = counts / counts.sum()
        log_probs = torch.log(probs.clamp(min=1e-12))
        self.initial_link_value = log_probs - log_probs.mean()
    
    def initial_link(self, num_samples):
        return self.initial_link_value.expand(num_samples, -1).clone()

    def psi(self, link):
        return F.softmax(link, dim=1)

    def predict_label(self, psi):
        return torch.argmax(psi, dim=1)

    def loss(self, psi, y):
        device = y.device

        y = y.squeeze().to(torch.long)
        log_psi = torch.log(psi.clamp(min=1e-12))
        return -log_psi[torch.arange(y.size(0), device=device, dtype=torch.long), y].mean()
    
    def gradient(self, psi):
        g = psi.clone()
        g[self.num_train_samples_range, self.y_train] -= 1.0
        return g

    def hessian_diagonal(self, psi):
        return psi*(1.0 - psi)

    def target_shape(self):
        return (self.num_classes,)

def build_normalized_adjacency_aggregator_cache(X, edge_index, edge_weight, degree_max):
    device, dtype = X.device, X.dtype

    aggregated_features = [X]

    if degree_max > 0:
        n = X.size(0)
        if edge_weight is None:
            e = edge_index.size(1)
            edge_weight = torch.ones(e, dtype=dtype, device=device)
        
        tail, head = edge_index
        deg = torch.zeros(n, dtype=dtype, device=device)
        deg.scatter_add_(0, tail, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
        A = SparseTensor(row=tail, col=head, value=edge_weight, sparse_sizes=(n, n))
        A_norm = A.set_value(edge_weight*deg_inv_sqrt[tail]*deg_inv_sqrt[head], layout='coo')
    
        for _ in range(degree_max):
            aggregated_features.append(A_norm.matmul(aggregated_features[-1]))

    return aggregated_features

def chebyshev_spectral_filter(aggregator_cache, params, aggregator_parity_separation, phi_out):
    device, dtype = aggregator_cache[0].device, aggregator_cache[0].dtype
    
    n = aggregator_cache[0].size(0)
    k = aggregator_cache[0].size(1)

    b = params[0]
    assert b.device == device and b.dtype == dtype
    filter_order_plus1 = b.numel()
    filter_order = filter_order_plus1 - 1

    if filter_order == 0:
        c = torch.stack([b[0]])
    elif filter_order == 1:
        c = torch.stack([b[0], b[1]])
    elif filter_order == 2:
        c = torch.stack([b[0] - b[2], b[1], 2.0*b[2]])
    elif filter_order == 3:
        c = torch.stack([b[0] - b[2], b[1] - 3.0*b[3], 2.0*b[2], 4.0*b[3]])
    elif filter_order == 4:
        c = torch.stack([b[0] - b[2] + b[4], b[1] - 3.0*b[3], 2.0*b[2] - 8.0*b[4], 4.0*b[3], 8.0*b[4]])
    elif filter_order == 5:
        c = torch.stack([b[0] - b[2] + b[4], b[1] - 3.0*b[3] + 5.0*b[5], 2.0*b[2] - 8.0*b[4], 4.0*b[3] - 20.0*b[5], 8.0*b[4], 16.0*b[5]])
    elif filter_order == 6:
        c = torch.stack([b[0] - b[2] + b[4] - b[6], b[1] - 3.0*b[3] + 5.0*b[5], 2.0*b[2] - 8.0*b[4] + 18.0*b[6], 4.0*b[3] - 20.0*b[5],
                         8.0*b[4] - 48.0*b[6], 16.0*b[5], 32.0*b[6]])
    elif filter_order == 7:
        c = torch.stack([b[0] - b[2] + b[4] - b[6], b[1] - 3.0*b[3] + 5.0*b[5] - 7.0*b[7], 2.0*b[2] - 8.0*b[4] + 18.0*b[6],
                         4.0*b[3] - 20.0*b[5] + 56.0*b[7], 8.0*b[4] - 48.0*b[6], 16.0*b[5] - 112.0*b[7], 32.0*b[6], 64.0*b[7]])
    elif filter_order == 8:
        c = torch.stack([b[0] - b[2] + b[4] - b[6] + b[8], b[1] - 3.0*b[3] + 5.0*b[5] - 7.0*b[7], 2.0*b[2] - 8.0*b[4] + 18.0*b[6] - 32.0*b[8],
                         4.0*b[3] - 20.0*b[5] + 56.0*b[7], 8.0*b[4] - 48.0*b[6] + 160.0*b[8], 16.0*b[5] - 112.0*b[7], 32.0*b[6] - 256.0*b[8],
                         64.0*b[7], 128.0*b[8]])
    else:
        T = torch.zeros(filter_order_plus1, filter_order_plus1, dtype=dtype, device=device)
        T[0, 0] = 1.0
        T[1, 1] = 1.0
        for m in range(1, filter_order):
            T[1:m+2, m+1] = 2.0*T[:m+1, m]
            T[:m, m+1] -= T[:m, m-1]
        c = T@b

    phi_out.zero_()

    if aggregator_parity_separation:
        phi_even = phi_out[:, :k]
        phi_odd = phi_out[:, k:]
        
        for m in range(filter_order_plus1):
            if m%2 == 0:
                phi_even[:] += c[m]*aggregator_cache[m]
            else:
                phi_odd[:] += c[m]*aggregator_cache[m]
    else:
        for m in range(filter_order_plus1):
            phi_out[:] += c[m]*aggregator_cache[m]

    return phi_out

class Iv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, t):
        device, dtype = t.device, t.dtype
    
        m = m.cpu().detach().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
        t = t.cpu().detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        
        I = torch.as_tensor(scipy.special.iv(m, t), dtype=dtype, device=device)
        I_plus = torch.as_tensor(scipy.special.iv(m + 1, t), dtype=dtype, device=device)
        I_minus = torch.as_tensor(scipy.special.iv(m - 1, t), dtype=dtype, device=device)
    
        m = torch.as_tensor(m, dtype=dtype, device=device)
        t = torch.as_tensor(t, dtype=dtype, device=device)
    
        
        ctx.save_for_backward(I, I_plus, I_minus)
        return I
    
    @staticmethod
    def backward(ctx, g):
        I, I_plus, I_minus = ctx.saved_tensors
        device, dtype = I.device, I.dtype
        
        return None, g*torch.as_tensor(0.5*(I_minus + I_plus), dtype=dtype, device=device)


class Ive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, t):
        device, dtype = t.device, t.dtype
    
        m = m.cpu().detach().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
        t = t.cpu().detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        
        I = torch.as_tensor(scipy.special.ive(m, t), dtype=dtype, device=device)
        I_plus = torch.as_tensor(scipy.special.ive(m + 1, t), dtype=dtype, device=device)
        I_minus = torch.as_tensor(scipy.special.ive(m - 1, t), dtype=dtype, device=device)

        m = torch.as_tensor(m, dtype=dtype, device=device)
        t = torch.as_tensor(t, dtype=dtype, device=device)

        t_sign = torch.sign(t)
        
        ctx.save_for_backward(I, I_plus, I_minus, t_sign)
        return I

    @staticmethod
    def backward(ctx, g):
        I, I_plus, I_minus, t_sign = ctx.saved_tensors
        device, dtype = I.device, I.dtype
        
        return None, g*torch.as_tensor(0.5*(I_minus + I_plus) - t_sign*I, dtype=dtype, device=device)
    

def _validate_tree_params(max_depth, min_child_weight, max_leaf_nodes, reg_lambda, min_split_loss):
    if max_depth is None and max_leaf_nodes is None: raise ValueError
    if max_depth is not None and max_depth < 1: raise ValueError
    if min_child_weight < 0.0: raise ValueError
    if max_leaf_nodes is not None and max_leaf_nodes < 2: raise ValueError
    if reg_lambda < 0.0: raise ValueError
    if min_split_loss < 0.0: raise ValueError

class SoftDecisionTreeLearner:
    def __init__(self, max_depth, min_child_weight, max_leaf_nodes, reg_lambda, min_split_loss, split_optimizer_cls, split_optimizer_kwargs, split_optimizer_epochs):
        self.max_depth = max_depth if max_depth is not None else INF
        self.min_child_weight = min_child_weight
        self.max_leaf_nodes = (
            max_leaf_nodes if max_leaf_nodes is not None else (1 << (max_depth - 1))
        )
        self.max_nodes = min(
            (1 << (max_depth + 1)) - 1 if max_depth is not None else INF,
            2 * max_leaf_nodes - 1 if max_leaf_nodes is not None else INF
        )
        self.reg_lambda = reg_lambda
        self.min_split_loss = min_split_loss
        
        self.split_optimizer_cls = split_optimizer_cls
        self.split_optimizer_kwargs = split_optimizer_kwargs
        self.split_optimizer_epochs = split_optimizer_epochs

    def fit(self, phi_T, g, h, split_optimizer_early_stopping_rounds, split_workspace):
        device, dtype = phi_T.device, phi_T.dtype
        
        # Local alias
        split_patience = split_optimizer_early_stopping_rounds
        split_ws = split_workspace

        self.target_shape = g.shape[1:]

        self.is_leaf = torch.empty(self.max_nodes, dtype=torch.bool, device="cpu")
        self.children = torch.empty(self.max_nodes, 2, dtype=torch.int32, device="cpu")
        self.feature = torch.empty(self.max_nodes, dtype=torch.int32, device="cpu")
        self.theta = torch.empty(self.max_nodes, dtype=dtype, device=device)
        self.tau = torch.empty(self.max_nodes, dtype=dtype, device=device)
        self.score = torch.empty(self.max_nodes, *self.target_shape, dtype=dtype, device=device)

        w_root = torch.ones(phi_T.size(1), dtype=dtype, device=device)
        G_root = w_root@g
        H_root = w_root@h
        
        split_candidate_queue = []
        gain, *split_children_stats = self._find_best_split(0, phi_T, g, h, w_root, G_root, H_root, split_patience, split_ws)
        heapq.heappush(split_candidate_queue, (-gain, 0, 0, split_children_stats))
        self.is_leaf[0] = False

        next_node_id = 1

        while split_candidate_queue and next_node_id + 1 < self.max_nodes:
            _, node_id, depth, split_children_stats = heapq.heappop(split_candidate_queue)
            
            for col, child_id in enumerate((next_node_id, next_node_id + 1)):
                self.children[node_id, col] = child_id

                _, G_child, H_child = split_children_stats[col]
                self.score[child_id] = -G_child/(H_child + self.reg_lambda + 1e-8)

                if depth >= self.max_depth - 1:
                    self.is_leaf[child_id] = True
                    continue

                gain, *stats = self._find_best_split(child_id, phi_T, g, h, *split_children_stats[col], split_patience, split_ws)
                if gain < self.min_split_loss:
                    self.is_leaf[child_id] = True
                    continue
                
                heapq.heappush(split_candidate_queue, (-gain, child_id, depth + 1, stats))
                self.is_leaf[child_id] = False

            next_node_id += 2

        for _, node_id, _, _ in split_candidate_queue:
            self.is_leaf[node_id] = True

    def _find_best_split(self, node_id, phi_T, g, h, w_node, G_node, H_node, split_patience, split_workspace):
        # Local alias
        split_ws = split_workspace

        k = phi_T.size(0)
        target_axes = tuple(range(-len(self.target_shape), 0))

        with torch.no_grad():
            split_ws.theta.normal_(0.0, 0.02)
            split_ws.tau_raw.normal_(0.0, 0.2)

        phi_T_detached = phi_T.detach()
        w_node_detached = w_node.detach()

        optimizer = self.split_optimizer_cls(
            [split_ws.theta, split_ws.tau_raw],
            **self.split_optimizer_kwargs
        )

        epochs_without_improvement = 0
        best_loss = float("inf")
            
        for _ in range(self.split_optimizer_epochs):
            for name, buf in split_ws._asdict().items():
                if name not in ["theta", "tau_raw"]:
                    buf.detach_()
            
            optimizer.zero_grad()

            # Apply softplus
            split_ws.tau.copy_(F.softplus(split_ws.tau_raw))
            split_ws.tau.add_(1e-8)

            loss_split = self._compute_loss_split(phi_T_detached, g, h, w_node_detached, G_node, H_node, split_ws, reduction="mean", selection="last")
            loss_split_sum = split_ws.loss_split.sum()
            
            loss_split_sum.backward()
            optimizer.step()

            epochs_without_improvement += 1
            
            loss_split_value = loss_split_sum.item()
            if loss_split_value < best_loss:
                best_loss = loss_split_value
                epochs_without_improvement = 0

                split_ws.best_theta.copy_(split_ws.theta.detach())
                split_ws.best_tau.copy_(split_ws.tau.detach())
            
            if split_patience is not None and epochs_without_improvement >= split_patience:
                break

        with torch.no_grad():
            loss_split = self._compute_loss_split(phi_T_detached, g, h, w_node_detached, G_node, H_node, split_ws, reduction="sum", selection="best")
            j = torch.argmin(loss_split)
    
            theta = split_ws.best_theta[j]
            tau = split_ws.best_tau[j]

        w_right = w_node*torch.sigmoid(tau*(phi_T[j] - theta))
        w_left = w_node - w_right

        with torch.no_grad():        
            H_left = w_left@h
            H_right = w_right@h
            
            if node_id != 0 and (H_left.sum() < self.min_child_weight or H_right.sum() < self.min_child_weight):
                return float("-inf"), None, None
    
            G_left = w_left@g
            G_right = w_right@g
            
            self.feature[node_id] = j
            self.theta[node_id] = theta
            self.tau[node_id] = tau
    
            node_loss = (-0.5*G_node**2/(H_node + self.reg_lambda + 1e-8)).sum(dim=target_axes)
            return (node_loss - loss_split[j]).item(), (w_left, G_left, H_left), (w_right, G_right, H_right)

    def _compute_loss_split(self, phi_T, g, h, w_node, G_node, H_node, split_workspace, reduction="mean", selection="last"):
        if reduction not in ("mean", "sum"):
            raise ValueError
        
        # Local alias
        split_ws = split_workspace
            
        if selection == "last":
            theta = split_ws.theta
            tau = split_ws.tau
        elif selection == "best":
            theta = split_ws.best_theta
            tau = split_ws.best_tau
        else:
            raise ValueError
        
        # Compute w_left, w_right
        split_ws.probit.copy_(phi_T)
        split_ws.probit.add_(theta[:, None], alpha=-1.0)
        split_ws.probit.mul_(tau[:, None])
        split_ws.probit.sigmoid_()
        split_ws.w_right.copy_(split_ws.probit)
        split_ws.w_right.mul_(w_node[None, :])
        split_ws.w_left.copy_(w_node[None, :])
        split_ws.w_left.add_(split_ws.w_right, alpha=-1.0)
        
        # Compute losses for the left and right child nodes, reduce along target axes, and sum to obtain the split loss
        for w_child, H_reg, loss_child in [(split_ws.w_left, split_ws.H_reg_left, split_ws.loss_left), (split_ws.w_right, split_ws.H_reg_right, split_ws.loss_right)]:
            target_axes = tuple(range(-len(self.target_shape), 0))
        
            H_reg.copy_(w_child@h)
            H_reg += self.reg_lambda + 1e-8
    
            loss_child.copy_(w_child@g)
            loss_child **= 2
            loss_child /= H_reg
            loss_child *= -0.5
            if reduction == "mean":
                loss_child /= w_child.sum(dim=1)[:, None].clamp(min=1e-12)

        split_ws.loss_split.copy_(split_ws.loss_left.sum(dim=target_axes))
        split_ws.loss_split.add_(split_ws.loss_right.sum(dim=target_axes))

        # Compute split loss
        return split_ws.loss_split
    
    def predict_link(self, phi_T):
        device, dtype = phi_T.device, phi_T.dtype
        
        target_ndim = len(self.target_shape)
        
        w_root = torch.ones(phi_T.size(1), dtype=dtype, device=device)
        
        split_candidate_queue = deque([(0, w_root)])
        link = torch.zeros(phi_T.size(1), *self.target_shape, dtype=dtype, device=device)
    
        while split_candidate_queue:
            node_id, w_node = split_candidate_queue.popleft()
            
            if self.is_leaf[node_id]:
                link += self.score[node_id]*w_node[(...,) + (None,) * target_ndim]
                continue
            
            w_right = w_node*torch.sigmoid(self.tau[node_id]*(phi_T[int(self.feature[node_id].item())] - self.theta[node_id]))
            w_left = w_node - w_right
            
            split_candidate_queue.extend([(int(self.children[node_id, 0]), w_left), (int(self.children[node_id, 1]), w_right)])

        return link

class GraphTreeBooster:
    _objectives = {
        "softmax_crossentropy": SoftmaxCrossEntropy
    }
    _aggregators = {
        "heat_kernel": chebyshev_spectral_filter,
        "chebyshev": chebyshev_spectral_filter
    }
    _aggregator_cache_builders = {
        "heat_kernel": build_normalized_adjacency_aggregator_cache,
        "chebyshev": build_normalized_adjacency_aggregator_cache
    }

    def __init__(self, objective, aggregator=None, aggregator_hparams=None, aggregator_parity_separation=False, max_depth=None, min_child_weight=1.0, max_leaf_nodes=None,
                 reg_lambda=1.0, min_split_loss=0.0, n_estimators=100, learning_rate=1.0,
                 tree_optimizer_cls=torch.optim.AdamW, tree_optimizer_kwargs=None, tree_optimizer_epochs=20,
                 split_optimizer_cls=torch.optim.AdamW, split_optimizer_kwargs=None, split_optimizer_epochs=50):
        _validate_tree_params(max_depth, min_child_weight, max_leaf_nodes, reg_lambda, min_split_loss)
        
        self.objective_cls = self._objectives[objective]
        self.aggregator_name = aggregator
        self.aggregator = self._aggregators.get(aggregator)
        self.aggregator_cache_builder = self._aggregator_cache_builders.get(aggregator)
        self.aggregator_hparams = aggregator_hparams
        self.aggregator_parity_separation = aggregator_parity_separation
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_leaf_nodes = max_leaf_nodes
        self.reg_lambda = reg_lambda
        self.min_split_loss = min_split_loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        self.tree_optimizer_cls = tree_optimizer_cls
        self.tree_optimizer_kwargs = tree_optimizer_kwargs
        self.tree_optimizer_epochs = tree_optimizer_epochs
        
        self.split_optimizer_cls = split_optimizer_cls
        self.split_optimizer_kwargs = split_optimizer_kwargs
        self.split_optimizer_epochs = split_optimizer_epochs
    
    def fit(self, X=None, y=None, edge_index=None, edge_weight=None, edge_label=None, train_idx=None, val_idx=None,
           tree_optimizer_early_stopping_rounds=None, split_optimizer_early_stopping_rounds=None, early_stopping_rounds=None, eval_metric=None):
        device, dtype = X.device, X.dtype

        # Local alias
        tree_patience = tree_optimizer_early_stopping_rounds
        split_patience = split_optimizer_early_stopping_rounds
        
        self.X = X
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
        n = X.size(0)
        n_train = train_idx.numel() if train_idx is not None else n
        k = X.size(1)
        k_ext = k*(2 if self.aggregator_parity_separation else 1)
        
        train_val_idx = train_idx if val_idx is None else torch.cat((train_idx, val_idx))
        
        y_train = y[train_idx] if train_idx is not None else y
        if val_idx is not None:
            y_val = y[val_idx]
        
        self.objective = self.objective_cls(y_train)
        target_shape = self.objective.target_shape()
        
        # Preallocate buffers (aggregated features)
        if self.aggregator is not None:
            phi_workspace = torch.empty(n, k_ext, dtype=dtype, device=device)
        
        # Preallocate buffers (split evaluation)
        SplitWS = namedtuple("SplitWS", ["theta", "tau_raw", "tau", "probit", "w_left", "w_right", "H_reg_left", "H_reg_right", "loss_left", "loss_right", "loss_split", "best_theta", "best_tau"])
        split_workspace = SplitWS(
            theta=torch.empty(k_ext, dtype=dtype, device=device, requires_grad=True),
            tau_raw=torch.empty(k_ext, dtype=dtype, device=device, requires_grad=True),
            tau=torch.empty(k_ext, dtype=dtype, device=device),
            probit=torch.empty(k_ext, n_train, dtype=dtype, device=device),
            w_left=torch.empty(k_ext, n_train, dtype=dtype, device=device),
            w_right=torch.empty(k_ext, n_train, dtype=dtype, device=device),
            H_reg_left=torch.empty(k_ext, *target_shape, dtype=dtype, device=device),
            H_reg_right=torch.empty(k_ext, *target_shape, dtype=dtype, device=device),
            loss_left=torch.empty(k_ext, *target_shape, dtype=dtype, device=device),
            loss_right=torch.empty(k_ext, *target_shape, dtype=dtype, device=device),
            loss_split=torch.empty(k_ext, dtype=dtype, device=device),
            best_theta=torch.empty(k_ext, dtype=dtype, device=device),
            best_tau=torch.empty(k_ext, dtype=dtype, device=device)
        )

        self.aggregator_cache = None if self.aggregator_cache_builder is None else \
            self.aggregator_cache_builder(X, edge_index, edge_weight, self.aggregator_hparams["filter_order"])
        
        if self.aggregator is not None:
            self.aggregator_params = []
            aggregator_params_candidates = []
        else:
            phi = X
            phi_T = phi.t()
            phi_train_val_T = phi_T.index_select(dim=1, index=train_val_idx)
            phi_train_T = phi_train_val_T[:, :n_train]
        
        self.learners = []
        learner_candidates = []

        link_train = self.objective.initial_link(train_idx.numel())
        if eval_metric is not None and val_idx is not None:
            link_val = self.objective.initial_link(val_idx.numel())

        if self.aggregator_name == "heat_kernel":
            filter_order_plus1 = self.aggregator_hparams["filter_order"] + 1
            beta_raw = torch.randn((), dtype=dtype, device=device)*0.1

            params_ = [beta_raw]
        elif self.aggregator_name == "chebyshev":
            filter_order_plus1 = self.aggregator_hparams["filter_order"] + 1
            b_logit = torch.tensor([1.0] + [0.0,]*self.aggregator_hparams["filter_order"], dtype=dtype, device=device)
            b_logit += 0.05 * torch.randn_like(b_logit) 
            
            params_ = [b_logit]

        if eval_metric is not None:
            metrics = [eval_metric] if isinstance(eval_metric, str) else list(eval_metric)
            
            if metrics[0] == "accuracy":
                best_metric_value = -1.0
            elif metrics[0] == "loss":
                best_metric_value = float("inf")
            rounds_without_improvement = 0

        for t in range(self.n_estimators):
            time_start = time.time()

            link_train_base = link_train.detach()
            response_train_base = self.objective.psi(link_train_base)
            g = self.objective.gradient(response_train_base)
            h = self.objective.hessian_diagonal(response_train_base)
            
            if self.aggregator is not None:
                if self.aggregator_name == "heat_kernel":                    
                    beta_raw.detach_()
                    beta_raw.requires_grad_()
                elif self.aggregator_name == "chebyshev":
                    b_logit.detach_()
                    b_logit.requires_grad_()
                
                optimizer = self.tree_optimizer_cls(
                    params_,
                    **self.tree_optimizer_kwargs
                )

                best_loss = float("inf")
                epochs_without_improvement = 0
                
                for _ in range(self.tree_optimizer_epochs):
                    phi_workspace.detach_()
                    
                    optimizer.zero_grad()
                    
                    if self.aggregator_name == "heat_kernel":
                        b = torch.empty(filter_order_plus1, dtype=dtype, device=device)
                        beta = F.softplus(beta_raw)
                        beta.add_(1e-8)
                        if self.aggregator_parity_separation:
                            b = Iv.apply(torch.arange(filter_order_plus1, dtype=torch.long, device=device), beta).to(device).clamp_min(1e-12)
                            b[::2] /= torch.cosh(beta)
                            b[1::2] /= torch.sinh(beta)
                        else:
                            b = Ive.apply(torch.arange(filter_order_plus1, dtype=torch.long, device=device), beta).to(device).clamp_min(1e-12)
                        b[1:] *= 2.0

                        params = [b]
                    elif self.aggregator_name == "chebyshev":
                        b = torch.empty(filter_order_plus1, dtype=dtype, device=device)
                        b = F.softmax(b_logit, dim=0)
                        params = [b]
                    
                    phi = self.aggregator(self.aggregator_cache, params, self.aggregator_parity_separation, phi_workspace)
                    phi_T = phi.t()
                    phi_train_T = phi_T.index_select(dim=1, index=train_idx)

                    weak_learner = SoftDecisionTreeLearner(self.max_depth, self.min_child_weight, self.max_leaf_nodes, self.reg_lambda, self.min_split_loss,
                                                    self.split_optimizer_cls, self.split_optimizer_kwargs, self.split_optimizer_epochs)
                    weak_learner.fit(phi_train_T, g, h, split_patience, split_workspace)
                    
                    link_train = link_train_base + self.learning_rate*weak_learner.predict_link(phi_train_T)
                    
                    response_train = self.objective.psi(link_train)
                    tree_loss = self.objective.loss(response_train, y_train)

                    tree_loss.backward()
                    optimizer.step()

                    epochs_without_improvement += 1
                    
                    tree_loss_value = tree_loss.item()
                    if tree_loss_value < best_loss:
                        best_loss = tree_loss_value
                        epochs_without_improvement = 0
                        aggregator_params_candidate = [p.detach() for p in params]
    
                    if tree_patience is not None and epochs_without_improvement >= tree_patience:
                        break

                aggregator_params_candidates.append(aggregator_params_candidate)

                with torch.no_grad():
                    phi = self.aggregator(self.aggregator_cache, aggregator_params_candidate, self.aggregator_parity_separation, phi_workspace)
                    phi_T = phi.t()
                    phi_train_val_T = phi_T.index_select(dim=1, index=train_val_idx)
                    phi_train_T = phi_train_val_T[:, :n_train]

            weak_learner = SoftDecisionTreeLearner(self.max_depth, self.min_child_weight, self.max_leaf_nodes, self.reg_lambda, self.min_split_loss,
                                            self.split_optimizer_cls, self.split_optimizer_kwargs, self.split_optimizer_epochs)
            weak_learner.fit(phi_train_T, g, h, split_patience, split_workspace)

            if t < self.n_estimators - 1 or eval_metric is not None:
                link_train = link_train_base + self.learning_rate*weak_learner.predict_link(phi_train_T)
            
            learner_candidates.append(weak_learner)

            time_end = time.time()
            
            if eval_metric is not None:
                with torch.no_grad():                
                    response_train = self.objective.psi(link_train)
                    
                    if val_idx is not None:
                        phi_val_T = phi_train_val_T[:, n_train:]
                        link_val += self.learning_rate*weak_learner.predict_link(phi_val_T)
                        response_val = self.objective.psi(link_val)

                    log_parts = [f"[{t}]"]
            
                    if "accuracy" in metrics:
                        y_pred_train = self.objective.predict_label(response_train)
                        train_acc = accuracy_score(y_train.cpu().numpy(), y_pred_train.cpu().numpy())
                        log_parts.append(f"Train Accuracy: {train_acc:.4f}")
                
                        if val_idx is not None:
                            y_pred_val = self.objective.predict_label(response_val)
                            val_acc = accuracy_score(y_val.cpu().numpy(), y_pred_val.cpu().numpy())
                            log_parts.append(f"Val Accuracy: {val_acc:.4f}")
                
                    if "loss" in metrics:
                        train_loss = self.objective.loss(response_train, y_train).item()
                        log_parts.append(f"Train Loss: {train_loss:.4f}")
                
                        if val_idx is not None:
                            val_loss = self.objective.loss(response_val, y_val).item()
                            log_parts.append(f"Val Loss: {val_loss:.4f}")

                    if "time" in metrics:
                        log_parts.append(f"Time: {time_end-time_start:.4f}")

                    print("\t".join(log_parts))
            
                rounds_without_improvement += 1
                
                metric_value, improvement_op = {"accuracy": (val_acc, operator.gt), "loss": (val_loss, operator.lt)}[metrics[0]]

                improved = improvement_op(metric_value, best_metric_value)
                if improved:
                    best_metric_value = metric_value
                    rounds_without_improvement = 0

                    if self.aggregator is not None:
                        self.aggregator_params += aggregator_params_candidates
                        aggregator_params_candidates = []
                
            if eval_metric is None or improved:
                self.learners += learner_candidates
                learner_candidates = []

            if early_stopping_rounds is not None and rounds_without_improvement >= early_stopping_rounds:
                break
    
    @torch.no_grad()
    def predict_link(self, X=None, edge_index=None, edge_weight=None, test_idx=None):
        if X is None and edge_index is None and edge_weight is None:
            if self.aggregator is not None:
                aggregator_cache = self.aggregator_cache
            
            X = self.X
            edge_index = self.edge_index
            edge_weight = self.edge_weight
        else:
            if self.aggregator_cache_builder is not None:
                aggregator_cache = self.aggregator_cache_builder(X, edge_index, edge_weight, self.aggregator_hparams["filter_order"])

        device, dtype = X.device, X.dtype

        n = X.size(0)
        n_test = test_idx.numel() if test_idx is not None else n
        k = X.size(1)
        k_ext = k*(2 if self.aggregator_parity_separation else 1)

        if self.aggregator is not None:
            phi_workspace = torch.empty(n, k_ext, dtype=dtype, device=device)
            link = self.objective.initial_link(n_test)
            for aggregator_params, weak_learner in zip(self.aggregator_params, self.learners):
                phi = self.aggregator(aggregator_cache, aggregator_params, self.aggregator_parity_separation, phi_workspace)
                phi_T = phi.t()
                phi_test_T = phi_T.index_select(dim=1, index=test_idx) if test_idx is not None else phi_T.contiguous()
                link += self.learning_rate * weak_learner.predict_link(phi_test_T)
        else:
            phi = X
            phi_T = phi.t()
            phi_test_T = phi_T.index_select(dim=1, index=test_idx) if test_idx is not None else phi_T.contiguous()
            link = self.objective.initial_link(n_test)
            for weak_learner in self.learners:
                link += self.learning_rate * weak_learner.predict_link(phi_test_T)
        
        return link
        
    @torch.no_grad()
    def predict(self, X=None, edge_index=None, edge_weight=None, test_idx=None):
        link = self.predict_link(X, edge_index, edge_weight, test_idx)
        return self.objective.psi(link)

    @torch.no_grad()
    def predict_label(self, X=None, edge_index=None, edge_weight=None, test_idx=None):
        response = self.predict(X, edge_index, edge_weight, test_idx)
        return self.objective.predict_label(response)

    @torch.no_grad()
    def evaluate(self, X=None, y=None, edge_index=None, edge_weight=None, edge_label=None, test_idx=None):
        response = self.predict(X, edge_index, edge_weight, test_idx)
        return self.objective.loss(response, y[test_idx]).item()
