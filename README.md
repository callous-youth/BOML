
# BOML - A Bilevel Optimization Library in Python for Multi-Task and Meta Learning
![Travis Status](https://travis-ci.com/dut-media-lab/BOML.svg?branch=master)
![codecov](https://codecov.io/gh/dut-media-lab/BOML/branch/master/graph/badge.svg)
![Documentation Status](https://readthedocs.org/projects/pybml/badge/?version=latest)
![license](https://img.shields.io/badge/license-MIT-000000.svg)
![Language](https://img.shields.io/github/languages/top/dut-media-lab/BOML)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

BOML is a modularized optimization library that unifies several ML algorithms into a common bilevel optimization framework. It provides interfaces to implement popular bilevel optimization algorithms, so that you could quickly build your own meta learning neural network and test its performance.

ReadMe.md file contains recommended instruction for training Maml-based and Meta-representation in few-shot learning field. It's flexible to build your own networks or use structures with attached documentation.
## Meta Learning and Multitask Learning

Meta learning works fairly well when facing incoming new tasks by learning an initialization with favorable generalization capability. And it also has good performance even provided with a small amount of training data available, which gives birth to new solutions to the few-shot learning problem.

![Hierarchically built strategies](https://latex.codecogs.com/gif.latex?\begin{equation}\min&space;_{\mathbf{x}}&space;F\left(\mathbf{x},\left\{\mathbf{y}^{i}\right\}_{i=1}^{N}\right),&space;\quad&space;\text&space;{&space;s.t.&space;}&space;\quad&space;\mathbf{y}^{i}&space;\in&space;\arg&space;\min&space;_{\mathbf{y}^{i}}&space;f\left(\mathbf{x},&space;\mathbf{y}^{i}\right),&space;i=1,&space;\cdots,&space;N\end{equation}{\color{DarkBlue}&space;}{\color{DarkBlue}&space;})

## Bilevel Structured strategies 
![Hierarchically built strategies](https://github.com/dut-media-lab/BOML/blob/master/figures/p1.png)

## Related Algorithms 
 - [Hyperparameter optimization with approximate gradient(Implicit HG)](https://arxiv.org/abs/1602.02355)
 - [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks(MAML)](https://arxiv.org/abs/1703.03400)
 - [On First-Order Meta-Learning Algorithms(FOMAML)](https://arxiv.org/abs/1803.02999)
 - [Bilevel Programming for Hyperparameter Optimization and Meta-Learning(Reverse HG)](http://export.arxiv.org/pdf/1806.04910)
 - [Truncated Back-propagation for Bilevel Optimization(Truncated Reverse HG)](https://arxiv.org/pdf/1810.10667.pdf)
 - [Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace(MTNet)](http://proceedings.mlr.press/v80/lee18a/lee18a.pdf)
 - [Meta-Learning with warped gradient Descent(Warp-Grad))](https://arxiv.org/abs/1909.00025)
 - [DARTS: Differentiable Architecture Search(DARTS)](https://arxiv.org/pdf/1806.09055.pdf)
 - [A Generic First-Order Algorithmic Framework for Bi-Level Programming Beyond Lower-Level Singleton(BDA)](https://arxiv.org/pdf/2006.04045.pdf)

## Package Structure
![Package Structure](https://github.com/dut-media-lab/BOML/blob/master/figures/p2.png)

## Documentation 

For more detailed information of basic function and construction process, please refer to our help page: [Help Documentation](https://bmlsoc.github.io/BOML/)

It's flexible to build your own networks or use structures in py_bm.networks. Scripts in the directory named train_script are useful for basic training process.

Here we give recommended settings for specific hyper paremeters to quickly test performance of popular algorithms.


