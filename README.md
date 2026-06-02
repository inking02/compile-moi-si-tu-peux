# compile-moi-si-tu-peux

IQCodeFest 2026 project focused on image-based anomaly detection using classical
and quantum machine learning experiments.

The goal is to classify aerial images with hybrid classical-quantum algorithms.
The project currently uses the NWPU dataset, which contains aerial views of
urban landmarks and infrastructure. This kind of pipeline can support use cases
such as disaster response, resource logistics, territorial monitoring, and
defense applications.

## Overview

The main workflow is implemented in [solution.ipynb](solution.ipynb) and follows a two-stage
classification pipeline.

First, the system decides whether an image is normal or anomalous. It uses a
ResNet18 feature extractor, reduces the feature space with PCA to 10
components and applies a One-Class QSVM to separate normal samples from
potential anomalies. This model is trained with a dataset with only "normal" data.

Then, when an anomaly is detected, the system classifies the precise anomaly
type. This stage uses a classical convolution network, a boson sampler layer that generates non-linearity, and a final linear layer to predict the anomaly class. This model is trained with only anomalies features.


## To run
To run the solution, please create a new environment and download the dependencies with this command. **Note that the code should be run with python version 3.12.0.**
```
pip install -r requirements.txt
```

