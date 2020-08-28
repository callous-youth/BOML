
# BOML - A Bilevel Optimization Library in Python for Multi-Task and Meta Learning
![Travis Status](https://travis-ci.com/dut-media-lab/BOML.svg?branch=master)
![codecov](https://codecov.io/gh/dut-media-lab/BOML/branch/master/graph/badge.svg)
![Documentation Status](https://readthedocs.org/projects/pybml/badge/?version=latest)
![license](https://img.shields.io/badge/license-MIT-000000.svg)
![Language](https://img.shields.io/github/languages/top/dut-media-lab/BOML)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)


BOML is a modularized optimization library that unifies several ML algorithms into a common bilevel optimization framework. It provides interfaces to implement popular bilevel optimization algorithms, so that you could quickly build your own meta learning neural network and test its performance.

ReadMe.md file contains recommended instruction for training meta-initialization based and meta-feature based approaches in few-shot learning field.

## Meta Learning and Bilevel Optimization

Meta-learning is the branch of machine learning that deals with the problem of \learning to learn" and has recently emerged as a potential learning paradigm that can gain experience over previous tasks and generalize that experience to unseen tasks proficiently.

We first present a general bilevel optimization paradigm to  unify different types of meta-learning approaches. Specifically, we define the meta dataset as $D=\{D^i\}_{i=1}^N$, where $D^i=D_{tr}^i\cup D_{val}^i$ is linked to the $i$-th task and $D^i_{tr}$ and $D^i_{val}$ respectively denote the training and validation sets. We denote the parameters of the base-learner as $y^i$ for the $i$-th task. Then the meta-learner can be thought of as a function that maps the dataset to the parameters of base-learner for new tasks, i.e., $y^i=\Psi(x,D}^i)$, where $x$ is the parameter of the meta-leaner and should be shared across tasks. With the above notations, we can formulate the general purpose of meta-learning tasks as the following bilevel optimization model:

\begin{equation}\min _{x} F\left(x,\left\{y^{i}\right\}_{i=1}^{N}\right), \quad \text { s.t. } \quad y^{i} \in \arg \min _{v^{i}} f\left(x, y^{i}\right), i=1, \cdots, N\end{equation}

where $f(\mathbf{x},\mathbf{y}^i)=\ell(\mathbf{x},\mathbf{y}^i,\mathcal{D}^i_{\mathtt{tr}})$ and $F(\mathbf{x},\{\mathbf{y}^i\}_{i=1}^N)=1/N\sum_{i=1}^N\ell(\mathbf{x},\mathbf{y}^i,\mathcal{D}^i_{\mathtt{val}})$ are called the lower and upper objectives, respectively. Here $\ell$ denotes task-specific loss functions (e.g., cross-entropy).

## Hierarchical structure of LL and UL strategies
Based on the above optimization details, BOML constructs six modules as basic frame-
work for our library: boml optimizer, load data, setup model, lower iter, upper iter
and optimizer. As the foremost optimization modules, the lower iter and upper iter
manage hierarchically built lower-level and upper-level strategies respectively to define the optimization model.

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
![Package Structure of BOML](https://github.com/dut-media-lab/BOML/blob/master/figures/p2.png)

## Documentation 

It's flexible to build your own networks or use structures in py_bm.networks. Scripts in the directory named test_script are useful for basic training process.
For more detailed information of basic function and construction process, please refer to our help page: [Help Documentation](https://bmlsoc.github.io/BOML/)

Here we give recommended settings for specific hyper paremeters to quickly test performance of popular algorithms.

## License

MIT License

Copyright (c) 2020 Yaohua Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



