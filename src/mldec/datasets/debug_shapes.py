import torch
from torch_geometric.loader import DataLoader
from mldec.datasets import reps_exp_rep_code_data, reps_toric_code_data

# Test repetition code dataset
print("=== Testing Repetition Code Dataset ===")
n_data = 10
repcode_dataset_config = {
    "code_size": 5,
    "repetitions": 5,
    "beta": 1
}
reps_data_tr, triv_tr, _ = reps_exp_rep_code_data.sample_dataset(n_data, repcode_dataset_config, None, seed=None)
reps_dataloader = DataLoader(reps_data_tr, batch_size=2, shuffle=False)

print(f"Number of repetition code samples: {len(reps_data_tr)}")
for i, batch in enumerate(reps_dataloader):
    print(f"Batch {i}:")
    print(f"  batch.x.shape: {batch.x.shape}")
    print(f"  batch.y.shape: {batch.y.shape}")
    print(f"  batch.edge_index.shape: {batch.edge_index.shape}")
    print(f"  batch.edge_attr.shape: {batch.edge_attr.shape}")
    print(f"  Number of nodes: {batch.x.shape[0]}")
    print(f"  Number of edges: {batch.edge_index.shape[1]}")
    print(f"  batch.batch.shape: {batch.batch.shape}")
    print(f"  batch.batch.max(): {batch.batch.max()}")
    print(f"  batch.batch.min(): {batch.batch.min()}")
    print(f"  batch.batch.unique(): {batch.batch.unique()}")
    break

# Test toric code dataset
print("\n=== Testing Toric Code Dataset ===")
toric_dataset_config = {
    "repetitions": 4,
    "code_size": 7,
    "p": 0.05,
    "beta": 1,
}
toric_data_tr, triv_tr, _, _ = reps_toric_code_data.sample_dataset(n_data, toric_dataset_config, None, seed=None)
toric_dataloader = DataLoader(toric_data_tr, batch_size=2, shuffle=False)

print(f"Number of toric code samples: {len(toric_data_tr)}")
for i, batch in enumerate(toric_dataloader):
    print(f"Batch {i}:")
    print(f"  batch.x.shape: {batch.x.shape}")
    print(f"  batch.y.shape: {batch.y.shape}")
    print(f"  batch.edge_index.shape: {batch.edge_index.shape}")
    print(f"  batch.edge_attr.shape: {batch.edge_attr.shape}")
    print(f"  Number of nodes: {batch.x.shape[0]}")
    print(f"  Number of edges: {batch.edge_index.shape[1]}")
    print(f"  batch.batch.shape: {batch.batch.shape}")
    print(f"  batch.batch.max(): {batch.batch.max()}")
    print(f"  batch.batch.min(): {batch.batch.min()}")
    print(f"  batch.batch.unique(): {batch.batch.unique()}")
    break

# Check individual samples
print("\n=== Individual Sample Analysis ===")
print("Repetition code sample 0:")
sample = reps_data_tr[0]
print(f"  x.shape: {sample.x.shape}")
print(f"  y.shape: {sample.y.shape}")
print(f"  edge_index.shape: {sample.edge_index.shape}")
print(f"  edge_attr.shape: {sample.edge_attr.shape}")

print("\nToric code sample 0:")
sample = toric_data_tr[0]
print(f"  x.shape: {sample.x.shape}")
print(f"  y.shape: {sample.y.shape}")
print(f"  edge_index.shape: {sample.edge_index.shape}")
print(f"  edge_attr.shape: {sample.edge_attr.shape}") 