# general notes:
 - train size doesnt matter as long as its big enough to sample one of each of 128 unique 

### Jan 28: Batch size 50, 1000 training, only good examples

Transformer:
 - more than 1 decoder layer hurts performance
 - lr > 0.01 mostly sucks
CNN: 
 - lr > 0.01 mostly sucks


 ### Feb 25: 2000 training, variable batch size only good examples
Transformer:
    - only 1 decoder layer is bad
    - lr < 0.005 is good (0.1 is bad)
    - again, batchsize and dropout are confounded by removing a lot of bad variables discovered in earlier runs

CNN:
 - kernel size 2 is bad
 - conv_channels 8 is bad
 - n_layers=2 is bad
 - dropout 0.05 _looks_ bad, but this is confounded by having a large number of experiments with dropout=0.05 fixed beforehand. same with batch_size=50
