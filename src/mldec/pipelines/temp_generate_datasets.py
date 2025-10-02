from mldec.datasets import single_round_code_data

ptrue = 0.001
var = 0.001
for (n, code_name) in [ (5, "fivequbit_code"), (7, "steane_code"),  (9, "toric_code")]:
    for var in [0, 0.001]:
        for beta in [1, 2, 3]:
            beta_dataset_config = {"p": ptrue, "var": var, "beta": beta, "dataset_module": code_name}
            X, Y, val_weights = single_round_code_data.create_dataset_training(n, beta_dataset_config)
            print(f"Generated dataset for {code_name} with p={ptrue}, var={var}, beta={beta}")
