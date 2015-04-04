# Features

Functionalities:
* convolutional layers
* max-pooling
* normalization layer
* random translations, as in [Flexible, High Performance Convolutional Neural Networks for Image Classification](http://ijcai.org/papers11/Papers/IJCAI11-210.pdf)
* random patches, as in [ImageNet Classification with Deep Convolutional Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* multinet, ie Multi-column deep convolutional network, [McDnn](http://arxiv.org/pdf/1202.2745.pdf)
* simple command-line network specification, as per notation in [Multi-column Deep Neural Networks for Image Classification](http://arxiv.org/pdf/1202.2745.pdf)
* pad-zeros possible for convolutional layer
* various activation functions available:
  * tanh
  * scaled tanh (1.7519 * tanh(2/3x) )
  * linear
  * sigmoid
  * relu
  * softmax
* fully-connected layers
* various loss layers available:
  * square loss
  * cross-entropy
  * multinomial cross-entropy (synonymous with multinomial logistic, etc)
* Q-learning
* Python wrapper

Example usage:
- intend to target 19 x 19 Go boards, eg something similar to [Clark and Storkey](http://arxiv.org/abs/1412.3409) or [Maddison, Huang, Sutskever and Silver](http://arxiv.org/abs/1412.6564)
  - obtained 36.3% test accuracy, on next move prediction task, using 33.6 million training examples from [kgsgo v2 dataset](https://github.com/hughperkins/kgsgo-dataset-preprocessor)
  - commandline used `./clconvolve1 dataset=kgsgoall netdef=32c5{z}-32c5{z}-32c5{z}-32c5{z}-32c5{z}-32c5{z}-500n-361n numepochs=3 learningrate=0.0001`
  - 3 epochs, 1.5 days per epoch, on an Amazon GPU instance, comprising half an NVidia GRID K520 GPU (about half as powerful as a GTX780)
- obtained 99.5% test accuracy on MNIST, using `netdef=rt2-8c5{padzeros}-mp2-16c5{padzeros}-mp3-150n-10n numepochs=20 multinet=6 learningrate=0.002`
  - epoch time 99.8 seconds, using an Amazon GPU instance, ie half an NVidia GRID K520 GPU (since we are learning 6 nets in parallel, so 16.6seconds per epoch per net)


