## Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning

This repository provides an implementation of the paper [Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning](https://arxiv.org/abs/2110.14049) accepted at [AISTATS 2022](https://aistats.org/aistats2022). We propose a **noise-reduced data valuation method, Beta Shapley**, which is a substantial generalization of Data Shapley. Beta Shapley outperforms state-of-the-art data valuation methods on several downstream ML tasks such as: 1) detecting mislabeled training data; 2) learning with subsamples; and 3) identifying points whose addition or removal have the largest positive or negative impact on the model.

<img src="./fig/beta_shapley_cifar100.png" width="600" alt="Beta Shapley on the CIFAR100 test dataset. Data points with a negative value are likely to be mislabeled.">

## Quick start

We provide a [notebook](notebook/Example_Covertype_dataset.ipynb) using the [Covertype dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html). It shows how to compute the Beta Shapley value and its application on several downstream ML tasks.

## Files

`betashap/ShapEngine.py`: main class for computing Beta-Shapley.

`betashap/data.py`: handles loading and preprocessing datasets.


