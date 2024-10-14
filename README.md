# NEMARL model

## DESCRIPTINON

![image](https://github.com/JasonKK1/NEMARLmodel/blob/main/nemarl.png)

In this project,we propose a NEMARL model to analyzing the evolution mechanism in networks as described in this paper.

## Installation

### Environment

* Python >=3.9

* Pytorch>=1.10

### Dependencies

1.Install Pytorch with the correct CUDA version

2.The following packages must also be installed:

* Numpy
* Pandas
* Pillow
* Scipy

## Files

### Crucial function：

**1.Algorithm:**

**Agent**: This file is responsible for configuring the agent, including defining its characteristics and behaviors necessary for the training process.

**Environment**: This file sets up the environment in which the agent operates. It defines the rules, rewards, and dynamics that the agent will interact with during training.

**Train**: This file orchestrates the training procedure, utilizing the agent and environment setups to facilitate the learning process. It manages the training loop, updates the agent based on its interactions with the environment, and evaluates its performance.

**2.GraphEmbedding:**

This directory  contains three distinct graph embedding methods: Struct2Vec, Graph Autoencoder (GAE), and GraphWaveMachinery.Each method is designed to delve into different aspects of graph representation.

**3.Common:**

In the common directory, there are two key components: arguments and evaluation. The arguments module is responsible for handling input parameters and configurations for the graph embedding methods. The evaluation module  includes classic metrics from complex network analysis, such as modularity, clustering coefficient.





## Questions

For any question about this program ,please contact

Email:202237576@mail.sdu.edu.cn



