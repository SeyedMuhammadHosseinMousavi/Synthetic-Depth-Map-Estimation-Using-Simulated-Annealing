# Synthetic Depth Data Generation Using Simulated Annealing (on Body Tracking Modality)
This repository contains several Python scripts for enhancing depth and normal maps using three different techniques: **Fuzzy Logic**, **ANFIS (Adaptive Neuro-Fuzzy Inference System)**, and **Simulated Annealing (SA)**. Each technique is applied to both **single images** and **videos**, with Simulated Annealing (SA) showing superior results compared to the other methods.

-The core of the algorithm relies on the **MiDaS model**, a pre-trained deep learning model designed for **depth estimation**, which generates depth maps from images or video frames.

## Overview

In this repository, we provide different approaches to enhance depth and normal maps using artificial intelligence techniques. Each approach aims to improve the visual and structural quality of depth and normal maps for various applications, such as 3D reconstruction, augmented reality, and more.

**Simulated Annealing (SA)** has been found to be more effective in achieving smoother and more coherent results compared to **Fuzzy Logic** and **ANFIS**, making it a better choice for both depth and normal map enhancement tasks.


https://github.com/user-attachments/assets/1446b621-c44c-4a43-a01c-c81f6958871d


## Content

The repository includes the following Python scripts, categorized by technique:

- **Fuzzy Logic**:
  - `0 Fuzzy Image to Depth.py` - Applies fuzzy logic to enhance depth maps for single images.
  - `0 Fuzzy Video to Depth.py` - Applies fuzzy logic to enhance depth maps for videos.

- **ANFIS (Adaptive Neuro-Fuzzy Inference System)**:
  - `1 ANFIS Image to Depth.py` - Applies ANFIS to enhance depth maps for single images.
  - `1 ANFIS Video to Depth.py` - Applies ANFIS to enhance depth maps for videos.

- **Simulated Annealing (SA)**:
  - `2 SA Image to Depth.py` - Applies simulated annealing to enhance depth maps for single images.
  - `2 SA Image to Normal.py` - Applies simulated annealing to enhance normal maps for single images.
  - `2 SA Video to Depth.py` - Applies simulated annealing to enhance depth maps for videos.
  - `2 SA Video to Normal.py` - Applies simulated annealing to enhance normal maps for videos.

## Techniques Explained

### 1. Fuzzy Logic

Fuzzy Logic uses fuzzy membership functions to adjust the depth values based on gradients. It applies simple fuzzy rules to regions with sharp or smooth transitions in the depth map.

- **Scripts**:
  - `0 Fuzzy Image to Depth.py`: Enhances depth maps in images by adjusting depth values based on fuzzy membership functions (low, medium, high depth).
  - `0 Fuzzy Video to Depth.py`: Enhances depth maps for video frames using fuzzy logic, frame by frame.

While Fuzzy Logic works well for simple depth adjustments, it may not handle complex transitions as effectively as more advanced techniques like Simulated Annealing (SA).

### 2. ANFIS (Adaptive Neuro-Fuzzy Inference System)

ANFIS combines fuzzy logic with neural networks to make predictions on the depth map. In this implementation, ANFIS adjusts depth values based on simple fuzzy rules with some neural network-like adaptation.

- **Scripts**:
  - `1 ANFIS Image to Depth.py`: Enhances depth maps for single images using ANFIS.
  - `1 ANFIS Video to Depth.py`: Enhances depth maps for videos using ANFIS.

ANFIS can provide more dynamic adjustments to depth maps compared to pure fuzzy logic, but it still lacks the flexibility and robustness of Simulated Annealing for high-quality depth map enhancement.

### 3. Simulated Annealing (SA)

Simulated Annealing (SA) is an optimization technique that works by minimizing sharpness or irregularities in the depth and normal maps. SA explores the solution space more effectively, ensuring a smoother and more realistic depth or normal map.

- **Scripts**:
  - `2 SA Image to Depth.py`: Enhances depth maps in single images using simulated annealing.
  - `2 SA Image to Normal.py`: Generates and enhances normal maps for single images using simulated annealing.
  - `2 SA Video to Depth.py`: Enhances depth maps for video frames using simulated annealing, frame by frame.
  - `2 SA Video to Normal.py`: Generates and enhances normal maps for video frames using simulated annealing.

Simulated Annealing outperforms both Fuzzy Logic and ANFIS, producing more coherent results in terms of smoothness and consistency in depth and normal maps. This technique is particularly effective for complex scenes where depth transitions need to be smooth and realistic.
![Synthetic Depth Data Generation Using Simulated Annealing](https://github.com/user-attachments/assets/6b84074c-c1a8-4e6b-b105-e069cd09d02d)

