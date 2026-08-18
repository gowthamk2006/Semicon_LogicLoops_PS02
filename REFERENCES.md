# SEMICON / DRIFT-SENSE
## References and Supporting Sources

This document lists the references used to support the image-processing,
augmentation, noise, and deep-learning methodology presented in the
SEMICON / DRIFT-SENSE project.

---

## 1. SEM / Semiconductor Pattern Inspection

KLA Corporation. Semiconductor inspection and metrology technologies.

Official Website:
https://www.kla.com/

**Purpose:**
Used as background reference for semiconductor inspection, metrology,
and SEM-based pattern analysis.

---

## 2. PyTorch Documentation

PyTorch. PyTorch Documentation.

Official Documentation:
https://pytorch.org/docs/stable/

**Purpose:**
Used as the implementation reference for the deep-learning model,
convolutional neural networks, tensor operations, model training,
and inference.

---

## 3. OpenCV Documentation

OpenCV. Open Source Computer Vision Library Documentation.

Official Documentation:
https://docs.opencv.org/

**Purpose:**
Used as the implementation reference for image loading, resizing,
image transformations, and computer-vision processing.

---

## 4. Image Data Augmentation

Shorten, C. and Khoshgoftaar, T. M. (2019).
*A survey on Image Data Augmentation for Deep Learning.*
Journal of Big Data, 6, 60.

**Purpose:**
Provides supporting background for the use of image transformations
and augmentation to improve robustness to variations in image
appearance.

---

## 5. Digital Image Processing and Noise

Gonzalez, R. C. and Woods, R. E.
*Digital Image Processing.*

**Purpose:**
Provides background on image degradation and noise in digital image
processing and supports the use of controlled noise perturbations
for robustness evaluation.

---

## 6. Augmentations Used in SEMICON / DRIFT-SENSE

The test and training data include controlled image variations
representing different acquisition and image-degradation conditions:

- Rotation
- Shear
- Scaling
- Drift
- Noise
- Blur
- Brightness/contrast variation

These transformations are used to evaluate the robustness of the
localization system under different image variations.

---

## 7. Project Methodology

The SEMICON / DRIFT-SENSE system uses a spatial-correlation-based
deep-learning approach to localize a target region from a reference
SEM image within a search SEM image.

The final Model D uses:

- CNN-based spatial feature extraction
- Spatial correlation
- Spatial score-map generation
- Soft-argmax coordinate decoding

The supporting documentation above provides the background for the
computer-vision, deep-learning, augmentation, and image-processing
components used in the project.
