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
### Training of T3-S1 model
The Vision Transformer model for self-supervised learning of three teacher networks and one student network that we use in the article is trained on three datasets.
```
python T3-S1 Network.py
```
## Evaluation
In this paper, two approaches are proposed for the evaluation of the multi-teacher knowledge distillation architecture. One method is to stitch together the feature dimensions in multiple model output tensors, and the other is to directly add the output tensors while keeping the feature dimensions unchanged.
### Evaluation of the DINO model
```
python linear.py
```
### Evaluation of the T2-S1 model
The Vision Transformer model for self-supervised learning of two teacher networks and one student network is evaluated on the dataset using the first feature-dimension combination method.
```
python T2-S1-cat.py
```
A Vision Transformer model for self-supervised learning of two teacher networks and one student network is evaluated on the dataset using the second feature dimension combination method.
```
python T2-S1-add.py
```
### Evaluation of the T3-S1 model
The Vision Transformer model for self-supervised learning of three teacher networks and one student network is evaluated on the dataset using the first feature-dimension combination method.
```
python T3-S1-cat.py
```
A Vision Transformer model for self-supervised learning of three teacher networks and one student network is evaluated on the dataset using the second feature dimension combination method.
```
python T3-S1-add.py
```
## Attention Image Visualization
We can see the self-attention of the [CLS] tokens of the DINO model on different heads of the last layer by running the code:
```
python visualize_attention.py
```
We can see the self-attention of the [CLS] tokens of the T2-S1 model used in the paper on different heads of the last layer by applying the following code:
```
python visualize_attention_t2.py
```

