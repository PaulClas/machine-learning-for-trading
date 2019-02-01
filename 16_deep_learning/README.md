# Chapter 16: Deep Learning

## How Deep Learning Works

- [Deep Learning](https://www.deeplearningbook.org/), Ian Goodfellow, Yoshua Bengio and Aaron Courville, MIT Press, 2016
- [Deep learning](https://www.nature.com/articles/nature14539), Yann LeCun, Yoshua Bengio, and Geoffrey Hinton, Nature 2015
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), Michael A. Nielsen, Determination Press, 2015
- [The Quest for Artificial Intelligence - A History of Ideas and Achievements](https://ai.stanford.edu/~nilsson/QAI/qai.pdf), Nils J. Nilsson, Cambridge University Press, 2010
- [One Hundred Year Study on Artificial Intelligence (AI100)](https://ai100.stanford.edu/)


### Backpropagation

- [Gradient Checking & Advanced Optimization](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization), Unsupervised Feature Learning and Deep Learning, Stanford University
- [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#momentum), Sebastian Ruder, 2016



## Popular Deep Learning libraries

- [Microsoft Cognitive Toolkit (CNTK)](https://github.com/Microsoft/CNTK)
- [Caffe](http://caffe.berkeleyvision.org/)
- [Thenao](http://www.deeplearning.net/software/theano/), developed at University of Montreal since 2007
- [Apache MXNet](https://mxnet.apache.org/), used by Amazon
- [Chainer](https://chainer.org/), developed by the Japanese company Preferred Networks
- [Torch](http://torch.ch/), uses Lua, basis for PyTorch
- [Deeplearning4J](https://deeplearning4j.org/), uses Java

### Leveraging GPU Optimization
- [Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](http://timdettmers.com/2018/11/05/which-gpu-for-deep-learning/), Tim Dettmers

### How to use Keras

- [A Full Hardware Guide to Deep Learning](http://timdettmers.com/2018/12/16/deep-learning-hardware-guide/), Tim Dettmers
- [Keras documentation](https://keras.io/)

### How to use Tensorboard

- [TensorBoard: Visualizing Learning](https://www.tensorflow.org/guide/summaries_and_tensorboard)

### How to use PyTorch

- [PyTorch Documentation](https://pytorch.org/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials)
- [PyTorch Ecosystem](https://pytorch.org/ecosystem)
    - [AllenNLP](https://allennlp.org/), state-of-the-art NLP platform developed by the Allen Institute for Artificial Intelligence
    - [Flair](https://github.com/zalandoresearch/flair),  simple framework for state-of-the-art NLP developed at Zalando
    - [fst.ai](http://www.fast.ai/), simplifies training NN using modern best practices; offers online training



    docker run -it -p 8889:8888 -v /home/stefan/projects/machine-learning-for-trading/17_convolutional_neural_nets:/cnn --name tensorflow tensorflow/tensorflow:latest-gpu-py3 bash

    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root