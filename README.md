# VLSKM: UAV Visual Localization Based on Sparse Keypoint Matching

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<!-- [TECHNICAL DETAIL NEEDED: Add any other relevant badges, e.g., arXiv link, PyTorch version, or IEEE Xplore link once published] -->

This repository contains the official implementation of **VLSKM** (Visual Localization via State-space Keypoint Matching), a sparse keypoint matching framework designed for reliable UAV visual localization in GNSS-denied environments. 

This work is currently submitted at *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*.

## 📝 Overview

UAV visual localization in GNSS-denied environments requires accurate geometric correspondence between a query UAV image and a georeferenced satellite or aerial reference image. This task is often compromised by severe cross-view disparities, in-plane rotations, spatial-coverage differences, and temporal appearance variations.

**VLSKM** addresses these challenges by improving correspondence modeling and geometric consistency. The framework consists of:
1. **Core Architecture:** Affine-steerable descriptors coupled with the DAMA Mixer to resolve severe cross-view disparities.
2. **Geometric Formulation:** Direct 2-D affine transformation yielding simultaneous latitude, longitude, and heading estimations without complex 3-D mapping.

By establishing reliable sparse correspondences, VLSKM estimates a robust homography. This geometric transformation is subsequently mapped to the georeferenced reference image to simultaneously estimate the UAV's **latitude, longitude, and heading angle**.

## 📊 Datasets

The framework is evaluated on Taipei-VLoc datasets to test both multi-temporal robustness and real-world applicability. 

### Taipei-VLoc
A synthetic cross-view UAV localization dataset designed to test varying spatial and temporal conditions.
* Features multiple query-image spatial coverage levels.
* Includes multiple satellite-image acquisition times to evaluate robustness to temporal appearance variation.
* **Download:** [Will be available soon]