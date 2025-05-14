### Toy problem experiments

#### FFN
 - /ffn-v0.*: Feedforward NN hyperparameter tuning runs
 - /ffn-v1.0: FNN data runs

#### CNN
  - cnn-v0/cnn_toy_problem_results_N2000_v4.csv: results for 1800 CNN runs, p=0.1, alpha=0.7, N=2000, batch size=250
  - cnn-v0/cnn_toy_problem_results_N100_v1.csv: results for 1000 CNN runs, p=0.1, alpha=0.7, N=100, batch size=100 
  - cnn-v0/cnn_toy_problem_only_good_examples*: hyperparameter tuning results

#### Transformer
 - xformer-v0/transformer_toy_problem_only_good_examples: hyperparameter tuning runs fortransformer
 - xformer-v0/transformer_toy_problem_results_N2000:results for 1350 transformer runs, p=0.1, alpha=0.7, N=2000

### Surface code experiments
Unless otherwise noted, all experiments here refer to L=3 rotated surface code with VAR=0.03, p=0.05

 - toric-code-(no)var-v1: First run for beta sweep of L=3 rotated surface code, but there was spurious seeding that caused the sampled training data to be the same across all runs. I am discarding these results as useless
 - toric-code-(no)var-v2: Corrected run for beta sweep, with and without variance in knob turning. v2.x: Additional samples at different beta values
  - toric-code-(no)var-v3-infinitedata: First run for beta sweep from v2, but with infinite data. Run failed, because using infinite data meant there was only 1 parameter update per epoch, down from n_train/batch_size updates. This was a mistake: when you have infinitely large minibatches, minibatching doesn't make sense anymore and you just have to increase epochs by a factor of n_batches 
  - toric-code-(no)var-v3.1-datasweep: This is a sweep along amount of training data, for two fixed beta values, with/without VAR

### FTQC experiments

 - reps-toric-code-exp-v0.x: hyperparameter tuning runs for GNN on repetitions=3 L=3 toric code
 - reps-toric-code-exp-v1.0?: All data runs for GNN with repetitions=5, L=3, beta=1,2,3, with various N. See main paper for noise model
