# CGQ-Net: Chest X-ray Report Omission Detection

Official implementation of **CGQ-Net (Category-Guided Query Network)** for category-level omission detection in chest X-ray reports. CGQ-Net performs category-specific visual and textual evidence retrieval and coverage-gated support–coverage reasoning, trained with controlled counterfactual supervision.

## Overview

This repository provides the implementation of **CGQ-Net (Category-Guided Query Network)**, a multimodal framework for detecting omitted findings in chest X-ray radiology reports.

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

The main-model implementation, training and evaluation configurations, category thresholds, and synthetic-input tests are available in this repository.

See [installation, data configuration, training, inference, and evaluation instructions](docs/USAGE.md). The checkpoint binary is distributed separately from this code upload.

## Citation

Citation information will be provided when the manuscript becomes publicly available.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
