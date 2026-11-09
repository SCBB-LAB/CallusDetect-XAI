# CallusDetect-XAI
### An explainable deep learning framework for automated callus detection and quality assessment in *Phyllostachys pubescens*

<p align="center">
  <strong>Explainable Artificial Intelligence | Deep Learning | Plant Tissue Culture | Callus Assessment</strong>
</p>

---

## 1. Overview

**CallusDetect-XAI** is an explainable deep learning-based framework developed for the automated detection and quality assessment of callus tissues in *Phyllostachys pubescens* (Moso bamboo).

Conventional callus assessment in plant tissue culture primarily relies on manual visual examination by experienced researchers. This process can be subjective, time-consuming, and difficult to standardize, particularly when evaluating large numbers of tissue culture samples.

To address these limitations, CallusDetect-XAI integrates an optimized YOLO11-based deep learning architecture with explainable artificial intelligence (XAI) techniques to provide rapid, objective, and interpretable callus assessment.

The framework incorporates architectural optimization, automated hyperparameter tuning, and visual explanation techniques to improve classification performance while providing insights into the morphological regions influencing model predictions.

## 2. Key Features

- **Automated Callus Assessment:** Deep learning-based identification of optimal and non-optimal callus tissues.
- **YOLO11 Architecture:** An optimized YOLO11-based framework for callus image analysis.
- **Architectural Enhancements:** Integration of C2PSA and C3K2 modules to improve feature extraction and representation.
- **Hyperparameter Optimization:** Optuna-based hyperparameter tuning for improved model performance.
- **Explainable AI:** Integration of Grad-CAM and Grad-CAM++ for visual interpretation of model predictions.
- **Morphological Assessment:** Identification of visual characteristics associated with callus quality.
- **Rapid Image Analysis:** Automated assessment to support high-throughput plant tissue culture screening.

## 3. Framework Architecture

The CallusDetect-XAI framework consists of four major components:

1. **Image Acquisition:** Collection of callus images under controlled laboratory conditions.
2. **Deep Learning Model:** Feature extraction and classification using the optimized YOLO11 architecture incorporating C2PSA and C3K2 modules.
3. **Hyperparameter Optimization:** Optuna-based optimization of model training parameters.
4. **Explainable AI:** Grad-CAM and Grad-CAM++ visualization to interpret the regions contributing to model predictions.

### Workflow

```text
          Callus Image Acquisition
                    |
                    v
             Image Dataset
                    |
                    v
          Image Preprocessing
                    |
                    v
         Optimized YOLO11 Model
                    |
          +---------+---------+
          |                   |
          v                   v
       C2PSA                C3K2
          |                   |
          +---------+---------+
                    |
                    v
          Feature Extraction
                    |
                    v
          Hyperparameter Tuning
                 (Optuna)
                    |
                    v
            Model Prediction
                    |
          +---------+---------+
          |                   |
          v                   v
      Optimal             Non-optimal
       Callus                Callus
          |                   |
          +---------+---------+
                    |
                    v
          Explainable AI Analysis
                    |
             +------+------+
             |             |
             v             v
          Grad-CAM      Grad-CAM++
             |             |
             +------+------+
                    |
                    v
        Interpretable Callus Assessment
```

## 4. Callus Classification

Callus samples are categorized into two classes based on their observable morphological characteristics.

| Category | Morphological characteristics |
|---|---|
| Optimal callus | Soft and friable texture, cream-to-light-yellow coloration, relatively homogeneous surface, and absence of visible necrotic regions. |
| Non-optimal callus | Browning or dark discoloration, watery or vitrified appearance, compact or hard morphology, and/or visible necrotic regions. |

These morphological characteristics provide the basis for image annotation and supervised model training.

## 5. Dataset Description

The dataset comprises experimentally acquired images of *Phyllostachys pubescens* callus tissues obtained under controlled laboratory conditions.

**Dataset characteristics:**

- Plant species: *Phyllostachys pubescens*
- Image acquisition: Controlled laboratory imaging setup using a Nikon camera.
- Dataset used for model development: 1,920 quality-controlled images.
- Classification categories: Optimal and non-optimal callus.
- Image source: Original experimental images generated during the study.
- Image annotations: Based on observable callus morphological characteristics.

The images were collected from different plant growth regulator, temperature, and light treatment combinations.

**Data availability:** The complete original image collection is approximately 49 GB. Therefore, the entire dataset is not included directly in this repository. A representative annotated image subset and additional dataset information will be made available through an appropriate data repository, subject to the publication and data-release arrangements.

## 6. Model Architecture

CallusDetect-XAI is developed using an optimized YOLO11-based architecture.

The framework incorporates the following components:

**C2PSA Module:** Introduced to enhance feature representation through attention-based feature processing.

**C3K2 Module:** Incorporated to improve feature extraction and computational efficiency.

**Optuna Hyperparameter Optimization:** Used to identify suitable training hyperparameters and improve model performance.

**Explainable AI Modules:** Grad-CAM and Grad-CAM++ are integrated to visualize the image regions contributing to model predictions.

## 7. Model Performance

The optimized CallusDetect-XAI framework achieved the following performance on the experimental evaluation dataset:

| Evaluation metric | Performance |
|---|---:|
| Accuracy | 97.57% |
| mAP@50 | 95.23% |

These results demonstrate the potential of the proposed framework for automated callus assessment.

*Note: The reported performance corresponds to the experimental evaluation described in the associated research study.*

## 8. Explainable Artificial Intelligence

To improve model interpretability, CallusDetect-XAI incorporates two gradient-based visualization approaches:

- **Grad-CAM:** Highlights image regions contributing to the model's predictions.
- **Grad-CAM++:** Provides refined class-discriminative localization of relevant image regions.

These visualization methods help investigate whether the model focuses on biologically meaningful morphological characteristics during callus assessment.

## 9. Repository Structure

The repository is organized as follows:

```text
CallusDetect-XAI/
│
├── README.md
│
├── dataset/
│   ├── sample_images/
│   └── annotations/
│
├── models/
│   ├── architecture
