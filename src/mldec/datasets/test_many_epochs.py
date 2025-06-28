import torch
from torch_geometric.loader import DataLoader
from mldec.datasets import reps_exp_rep_code_data, reps_toric_code_data
from mldec.models import initialize

# Test repetition code dataset with many epochs
print("=== Testing Repetition Code Dataset with Many Epochs ===")
n_data = 2048
repcode_dataset_config = {
    "code_size": 5,
    "repetitions": 5,
    "beta": 1
}
reps_data_tr, triv_tr, _ = reps_exp_rep_code_data.sample_dataset(n_data, repcode_dataset_config, None, seed=None)
reps_dataloader = DataLoader(reps_data_tr, batch_size=4, shuffle=True)

# Create a simple model for testing
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

print(f"Testing repetition code training for many epochs...")
print(f"Dataset size: {len(reps_data_tr)}")
print(f"Batch size: 4")
print(f"Batches per epoch: {len(reps_dataloader)}")

max_epochs = 100
for epoch in range(max_epochs):
    print(f"Epoch {epoch+1}/{max_epochs}")
    for batch_idx, batch in enumerate(reps_dataloader):
        try:
            # Try the training step
            correct_predictions, loss = reps_model_wrapper.training_step(batch, reps_optimizer, reps_criterion)
            if batch_idx == 0:  # Only print first batch of each epoch to avoid spam
                print(f"  Batch {batch_idx}: x.shape={batch.x.shape}, y.shape={batch.y.shape}, loss={loss.item():.4f}")
        except Exception as e:
            print(f"  ERROR at epoch {epoch+1}, batch {batch_idx}:")
            print(f"    batch.x.shape: {batch.x.shape}")
            print(f"    batch.y.shape: {batch.y.shape}")
            print(f"    batch.batch.shape: {batch.batch.shape}")
            print(f"    batch.batch.unique(): {batch.batch.unique()}")
            print(f"    Error: {e}")
            print(f"    Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    if (epoch + 1) % 10 == 0:
        print(f"  Completed {epoch+1} epochs successfully")

print("All epochs completed successfully!") 