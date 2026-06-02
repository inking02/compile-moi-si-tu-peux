# compile-moi-si-tu-peux

IQCodeFest 2026 project focused on image-based anomaly detection using classical
and quantum machine learning experiments.

The goal is to classify aerial images with hybrid classical-quantum algorithms.
The project currently uses the NWPU dataset, which contains aerial views of
urban landmarks and infrastructure. This kind of pipeline can support use cases
such as disaster response, resource logistics, territorial monitoring, and
defense applications.

## Overview

The project pipeline follows three main steps:

- `data/NWPU`: loads the dataset and generates anomaly labels.
- `denoiser`: simulates haze and optionally applies convolutional denoising.
- `anomalies_detection`: performs QSVM-based anomaly detection and classification.

In short, the system takes a potentially degraded aerial image, optionally
denoises it, extracts visual features, and runs two classification stages: one
to detect whether an anomaly is present, and another to identify the anomaly
type.
