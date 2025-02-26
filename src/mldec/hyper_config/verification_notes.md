# general notes:
 - train size doesnt matter as long as its big enough to sample one of each of 128 unique 

### Jan 28: Batch size 50, 1000 training, only good examples

Transformer:
 - more than 1 decoder layer hurts performance
 - lr > 0.01 mostly sucks
CNN: 
 - lr > 0.01 mostly sucks

 ### Feb 25: Batch size 100, 1000 training, only good examples
Transformer:
    TODO
CNN:
 - kernel size 2 is bad
 - conv_channels 8 is bad

 ### Feb 25: 2000 training, variable batch size only good examples

to check: 
 - last cnn run with batch sizes, 
 - last two transformer runs with dropouts and batch sizes