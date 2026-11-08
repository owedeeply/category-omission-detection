# CGQ-Net: Chest X-ray Report Omission Detection

**Official implementation of "Detecting Omitted Findings in Chest X-ray Reports via Category-Specific Support–Coverage Modeling".**

## Overview

This repository provides the implementation of **CGQ-Net (Coverage-Gated Query Network)**, a multimodal framework for detecting omitted findings in chest X-ray radiology reports.

Unlike conventional report evaluation methods that primarily assess factual correctness, our approach focuses on identifying clinically relevant findings supported by chest X-ray images but not adequately documented in their accompanying reports.

## Method

CGQ-Net introduces a category-specific support–coverage modeling framework that integrates:

- **Category-specific queries** to extract finding-level representations from chest X-ray images and radiology reports.
- **Coverage-aware gating** to identify visually supported findings that remain insufficiently covered in the report.
- **Controlled counterfactual supervision** to learn omission patterns through targeted removal of finding-related report content.
- **Multitask learning** to jointly optimize omission detection, visual-support estimation, and report-coverage estimation.

The framework performs category-level omission detection across nine clinically relevant finding categories.

## Dataset

The experiments are based on the MIMIC-CXR and MIMIC-CXR-JPG datasets.

Access to the original datasets is subject to PhysioNet credentialing and data-use requirements. Restricted source data are not redistributed through this repository.

## Code Availability

The implementation, configuration files, and instructions for training and evaluation are being organized for release.

## Citation

Citation information will be provided when the manuscript becomes publicly available.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
