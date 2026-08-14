2D Anthropometric Estimation Framework
This project introduces a novel, efficient framework for human body measurement estimation. By utilizing a 2D silhouette-based approach, this system eliminates the computational overhead associated with traditional 3D parametric model fitting, resulting in significantly faster training cycles and improved resource efficiency.

Overview
Traditional body measurement systems often rely on heavy 3D human mesh recovery (HMR) pipelines, which require iterative and computationally expensive optimization. This framework shifts the paradigm by regressing physical body measurements, such as limb lengths and circumference parameters, directly from 2D silhouette deformation patterns.

Key Features
Training Efficiency: Bypasses iterative 3D mesh optimization to streamline the training pipeline.

Lightweight Architecture: Utilizes an optimized neural network architecture designed for high throughput.

High Precision: Achieves measurement accuracy comparable to leading 3D parametric models, with a reliable standard deviation error of 2.1 mm.

This is of low BMI:
<img width="1112" height="678" alt="1" src="https://github.com/user-attachments/assets/ef70d349-3a0e-40e0-bef4-954aeb96127f" />

This is of High BMI:
<img width="972" height="762" alt="image" src="https://github.com/user-attachments/assets/761ccc0f-91ac-4adc-977c-3ad6b582a31e" />


Getting Started
This repository contains the training and inference pipelines required to replicate the results. Ensure you have the necessary environment configured as per the provided requirements.txt before executing the training scripts.

License:
This project is for research and development purposes as part of an independent study.
