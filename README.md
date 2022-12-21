# Tn-S1-Network
Vision Transformer Model of Multi teacher Knowledge Distillation and Self supervised Learning.
## Training
### Documentation
Please install PyTorch and download cifar-10 dataset, Fashion MNIST dataset, and ImageNet dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. 
### Training of DINO model
Here we are using the source code of the DINO model trained on three datasets.
```
python main_dino.py
```
### Training of T2-S1 model
The Vision Transformer model for self-supervised learning of two teacher networks and one student network that we use in the article is trained on three datasets.
```
python T2-S1 Network.py
```
