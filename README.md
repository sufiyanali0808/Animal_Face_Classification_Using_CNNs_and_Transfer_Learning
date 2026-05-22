# Animal Faces Multi-Class Classification Using Custom CNNs

This project implements and evaluates custom Convolutional Neural Networks (CNNs) in PyTorch to classify animal faces into three distinct categories: **Cats, Dogs, and Wildlife**. The primary focus of this repository is exploring how architectural depth and data augmentation impact training dynamics, validation performance, and overfitting.

---

## 1.  Hardware Used
Training performed on NVIDIA T4 GPU using PyTorch.

---

## 2. Dataset
The project utilizes the **Animal Faces Dataset**, which consists of 16,130 high-quality images at 512×512 resolution each class providing about 5000 images. The classes are : 
* **Cat**
* **Dog**
* **Wildlife**

All images are resized to 224 * 224  pixels during the preprocessing pipeline to match standard computer vision inputs.

---

## 3. Network Architectures & Experiments

We designed and tracked three distinct configurations to analyze structural and generalization trade-offs:

1. **3-Layer CNN (Without Augmentation)**
   * **Structure:** 3 Sequential Convolutional layers interleaved with ReLU activations and MaxPool2d layers, culminating in a Fully Connected classification head.
   * **Data Strategy:** Standard resizing and normalization only.

2. **3-Layer CNN (With Augmentation)**
   * **Structure:** Identical to the initial 3-layer architecture.
   * **Data Strategy:** Integrated random horizontal flips, rotations, and color jitter to artificially expand the training variance.

3. **4-Layer CNN (With Augmentation)**
   * **Structure:** Features an additional 4th Convolutional layer block to capture higher-level spatial abstractions, maintaining the same data augmentation pipeline.

---

## 4. Model Training Configuration & Pipeline

The models are compiled and trained using standard deep learning components optimized for multi-class classification tasks. The network utilizes **CrossEntropyLoss** as the loss function, which inherently combines a `LogSoftmax` layer and negative log-likelihood loss to measure the performance of the classification output. Optimization is handled via the **Adam optimizer**, configured with a learning rate of `0.0001` to ensure stable and adaptive gradient updates. To efficiently manage the pipeline, the **Animal Faces Dataset** is split into training, validation and test sets. These datasets are then wrapped in **DataLoader** instances with a batch size of `16` and `shuffle=True` for the training subset, enabling parallelized batch processing, automatic data shuffling to prevent sequence bias, and streamlined memory management during training loops.

---

## 5. Experimental Results

The models were evaluated based on their final validation accuracies:

| Architecture Configuration | Data Augmentation | Evaluation Accuracy |
| :--- | :--- | :--- |
| **3-Layer CNN** | No | **99.786%** |
| **3-Layer CNN** | Yes | **95.5185%** |
| **4-Layer CNN** | Yes | **96.6168%** |

---

## 6. Key Technical Insights & Overfitting Analysis

While a casual look at the surface numbers might suggest that the unaugmented 3-Layer model is the best performer, a deeper look at the optimization curves reveals critical nuances regarding **generalization vs. overfitting**:

### 1. The Paradox of the 99.786% Accuracy (Overfitting)
* **What Happened:** The 3-Layer CNN *without* augmentation quickly memorized the raw pixel patterns, background textures, and lighting conditions specific to the training set. 
* **When & Why:** This extreme convergence happens early when the model has enough capacity to memorize a limited variation of images. Because the evaluation dataset shares highly similar environment contexts, the validation score peaked exceptionally high. However, this is the case of **overfitting**. The model lacks generalizability and may exhibit weaker robustness on real-world unseen samples.

  <img width="1678" height="594" alt="3_CNN_layer" src="https://github.com/user-attachments/assets/76bbaccf-34cb-4d80-8fe9-3e69e89a6126" />


### 2. Regularization via Data Augmentation
* **The Drop in Accuracy:** Introducing random transformations purposefully made the training phase harder. This dropped the raw evaluation accuracy down to **95.5185%**.
* **Why this is a Win:** This drop is actually highly beneficial. Augmentation acts as a regularizer, preventing the network from relying on static pixel coordinates or background features. It forces the model to learn invariant geometric structures (like the actual shape of ears, eyes, and snouts), making it vastly more robust.

  <img width="1317" height="459" alt="image" src="https://github.com/user-attachments/assets/23ec26dc-6021-4db6-a62f-83f56f01e782" />


### 3. Structural Capacity Scaling (Moving to 4 Layers)
* **The Boost:** By introducing a 4th Convolutional block alongside data augmentation, the model performance reclaimed ground up to **96.6168%**.
* **The Reason:** Data augmentation increases the complexity and variance of the data distribution. A 3-layer network struggles to fit this harder data well without underperforming. Adding a 4th layer provides the network with more parameters and higher-level abstraction fields to learn these complex, augmented representations cleanly without reverting to cheap memorization.

  <img width="1676" height="589" alt="4_CNN_layer_with_aug" src="https://github.com/user-attachments/assets/7270cbf5-b878-4c64-b497-024fa73c991c" />


---

### 7. Benchmark Comparison with Deep Transfer Learning (ResNet)
To establish a performance baseline against industry-standard architectures, a pre-trained **ResNet** model was fine-tuned on the same dataset using identical data split conditions. The ResNet architecture achieved a final evaluation accuracy of **99.5926%**. While our unaugmented 3-layer CNN reached a numerically comparable peak (99.786%), the optimization trajectory of the deep ResNet model demonstrated true feature generalization rather than localized dataset memorization. Leveraging pre-trained residual blocks allowed the network to utilize robust, low-level primitive features (edges, textures, and structural gradients) learned from massive generalized image bases. This benchmark concludes that while lightweight custom architectures can be hyper-optimized to fit a specific distribution close to 100%, scaling to deep residual architectures or incorporating targeted regularization remains essential for deploying reliable, non-overfitting vision models in production scenarios.

<img width="1683" height="595" alt="Screenshot_20260522_175136" src="https://github.com/user-attachments/assets/43537762-47c5-4e95-8950-ef89284a6f62" />


---

## 8. Future Work
1. Confusion Matrix
2. Vision Transformer
3. Hyperparameter Tuning

---
