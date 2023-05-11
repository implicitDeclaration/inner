 
# Introduction

Code to the paper "Interpretability based Neural Network Repair".

The implementation of this project is based on pytorch 1.13 with python 3.7.0 


# Useage
put your dataset, e.g., cifar10 in the folder of './dataset/cifar10', so the code can find the data
### 1. Model generation
Train models with certain architecture and dataset, specify the certain repair scenario and it will be trained as well:

```
python main.py --mode train --arch vgg13_dense --set cifar10 --save_dir ./checkpoints/cifar10 --rep_type bd
```
This will generate folders containing the model weights, other operations depend on the args to determine the path of weights. 

### 2. Train model probes

You must specify the number of samples and target task to train the probe

```
python main.py --mode probe  --probe_train_num 1000 --arch vgg13_dense --set cifar10 --rep_type bd --save_dir ./checkpoints/cifar10
```
### 3. repair model
You can specify the number of samples and layers for repair  
```
python main.py --mode rep --probe_train_num 1000 --arch vgg13_dense --set cifar10 --rep_type bd --rep_layer_num 3 --repair_sample_num 1000 --save_dir ./checkpoints/cifar10 --rep_dir ./checkpoints/repaired/
```

### 4. test repair
Test will generate a log file containing the results
```
python  main.py --mode test  --probe_train_num 1000 --arch vgg13_dense --set cifar10 --rep_type bd --rep_layer_num 3 --repair_sample_num 1000 --save_dir ./checkpoints/cifar10 --rep_dir ./checkpoints/repaired/
```













 
