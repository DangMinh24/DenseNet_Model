# DenseNet_Model

Simple implementation of DenseNet structure on ImageNet dataset.

Note: This implementation is according to the original paper. Although not sure exactly how authors's network was built, I tried to represent the simplest construction according to what I understand from the paper.
I tried to avoid ZeroPadding layer, to focus only on Dense Block, Transition Block and BottleNeck (to simplify and easy to follow shape of input and output in each layer)


Ref: [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

Ref_model: [link](https://github.com/flyyufelix/DenseNet-Keras)