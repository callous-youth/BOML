
# BOML - A Bilevel Optimization Library in Python for Multi-Task and Meta Learning
![Travis Status](https://travis-ci.com/bmlsoc/BOML.svg?branch=master)
![license](https://img.shields.io/badge/license-MIT-000000.svg)
![Documentation Status](https://readthedocs.org/projects/pybml/badge/?version=latest)
![Language](https://img.shields.io/badge/language-Python-brightgreen.svg)



BOML is a modularized optimization library that unifies several ML algorithms into a common bilevel optimization framework. It provides interfaces to implement popular bilevel optimization algorithms, so that you could quickly build your own meta learning neural network and test its performance.

ReadMe.md file contains recommended instruction for training Maml-based and Meta-representation in few-shot learning field. It's flexible to build your own networks or use structures with attached documentation.
## Meta Learning and Multitask Learning

Meta learning works fairly well when facing incoming new tasks by learning an initialization with favorable generalization capability. And it also has good performance even provided with a small amount of training data available, which gives birth to new solutions to the few-shot learning problem.

In few-shot learning problems, we consider N-way K-shot classification tasks. For each iteration, $M$ batches of tasks $\{T_{i=1}^{M}\}$ are randomly sampled from distribution $P(T)$ associated with split datasets $D_{i}^{tr}$, $D_{i}^{val}$. Both datasets take the form of $D_{i}=\{(x_{i}^{k}, y_{i}^{k})\}^{K}_{k=1}$. And each K samples $x\in\mathcal{X}$ from the same class are bounded to specific label $y\in\mathcal{Y}$, which can be formulated as a mapping $h(x):\mathcal{X}\rightarrow \mathcal{Y}$. Meta learning aims to improve the model’s classification ability on new instances within N classes after training using sampled tasks. 

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


