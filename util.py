import torch
from torch_sparse import SparseTensor

def standardize(X=None, edge_index=None, train_mask=None, val_mask=None, test_mask=None, max_dist=0):
    n = X.size(0)
    if max_dist > 0:
        tail, head = edge_index
    
        edge_keep = None
        if val_mask is not None:
            edge_keep = ~(val_mask[tail] | val_mask[head])
        if test_mask is not None:
            mask_test = ~(test_mask[tail] | test_mask[head])
            edge_keep = mask_test if edge_keep is None else edge_keep & mask_test
    
        if edge_keep is not None:
            tail = tail[edge_keep]
            head = head[edge_keep]
        
        A_keep = SparseTensor(row=tail, col=head, sparse_sizes=(n, n))
    
        keep_mask = train_mask.to(torch.float32).unsqueeze(1).clone()
        for _ in range(max_dist):
            keep_mask += A_keep.matmul(keep_mask)
        keep_mask = keep_mask.squeeze(1) > 0
    else:
        keep_mask = train_mask
    
    X_keep = X[keep_mask]

    nonzero_std_mask = X_keep.std(dim=0, correction=0) > 0.0
    X = X[:, nonzero_std_mask]
    X_keep = X_keep[:, nonzero_std_mask]
    
    mean = X_keep.mean(dim=0)
    std = X_keep.std(dim=0, correction=0)
    return (X - mean) / std