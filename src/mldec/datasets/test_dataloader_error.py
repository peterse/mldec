import torch
from torch_geometric.loader import DataLoader
from mldec.datasets import reps_exp_rep_code_data, reps_toric_code_data

# Test repetition code dataset with actual training step
print("=== Testing Repetition Code Dataset with Training Step ===")
n_data = 10
repcode_dataset_config = {
    "code_size": 5,
    "repetitions": 5,
    "beta": 1
}
reps_data_tr, triv_tr, _ = reps_exp_rep_code_data.sample_dataset(n_data, repcode_dataset_config, None, seed=None)
reps_dataloader = DataLoader(reps_data_tr, batch_size=2, shuffle=False)

# Create a simple model for testing
from mldec.models import initialize
dummy_model_config = {
    "model": "gnn",
    "gcn_depth": 2,
    "gcn_min": 16,
    "mlp_depth": 3,
    "mlp_max": 8,
    "patience": 100,
    "opt": "adam",
    "input_dim": 2,  # repetition code has 2 features
    "output_dim": 1,
    "lr": 0.003,
    "dropout": 0.05,
}

reps_model_wrapper = initialize.initialize_model(dummy_model_config)
reps_criterion = torch.nn.BCEWithLogitsLoss()
reps_optimizer = torch.optim.Adam(reps_model_wrapper.model.parameters(), lr=0.003)

print("Testing repetition code training step...")
for i, batch in enumerate(reps_dataloader):
    print(f"Batch {i}:")
    print(f"  batch.x.shape: {batch.x.shape}")
    print(f"  batch.y.shape: {batch.y.shape}")
    print(f"  batch.batch.shape: {batch.batch.shape}")
    print(f"  batch.batch.unique(): {batch.batch.unique()}")
    
    try:
        # Try the training step
        reps_model_wrapper.training_step(batch, reps_optimizer, reps_criterion)
        print("  Training step succeeded!")
    except Exception as e:
        print(f"  Training step failed with error: {e}")
        print(f"  Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    break

print("\n=== Testing Toric Code Dataset with Training Step ===")
toric_dataset_config = {
    "repetitions": 4,
    "code_size": 7,
    "p": 0.05,
    "beta": 1,
}
toric_data_tr, triv_tr, _, _ = reps_toric_code_data.sample_dataset(n_data, toric_dataset_config, None, seed=None)
toric_dataloader = DataLoader(toric_data_tr, batch_size=2, shuffle=False)

# Create a simple model for testing
toric_model_config = dummy_model_config.copy()
toric_model_config["input_dim"] = 5  # toric code has 5 features

toric_model_wrapper = initialize.initialize_model(toric_model_config)
toric_criterion = torch.nn.BCEWithLogitsLoss()
toric_optimizer = torch.optim.Adam(toric_model_wrapper.model.parameters(), lr=0.003)

print("Testing toric code training step...")
for i, batch in enumerate(toric_dataloader):
    print(f"Batch {i}:")
    print(f"  batch.x.shape: {batch.x.shape}")
    print(f"  batch.y.shape: {batch.y.shape}")
    print(f"  batch.batch.shape: {batch.batch.shape}")
    print(f"  batch.batch.unique(): {batch.batch.unique()}")
    
    try:
        # Try the training step
        toric_model_wrapper.training_step(batch, toric_optimizer, toric_criterion)
        print("  Training step succeeded!")
    except Exception as e:
        print(f"  Training step failed with error: {e}")
        print(f"  Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    break 