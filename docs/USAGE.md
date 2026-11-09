# CGQ-Net

Official implementation of **CGQ-Net (Category-Guided Query Network)** for category-level omission detection in chest X-ray reports. CGQ-Net performs category-specific visual and textual evidence retrieval and coverage-gated support–coverage reasoning, trained with controlled counterfactual supervision.

This repository provides the main model, training, inference, threshold fitting, and benchmark evaluation for *Detecting Omitted Findings in Chest X-ray Reports via Category-Specific Support–Coverage Modeling*.

## Installation

Use Python 3.10 or 3.11 in a dedicated environment, with a PyTorch build appropriate for your CUDA installation or CPU. The dependency versions in `requirements.txt` match the recorded experiment environment.

```bash
python -m pip install -r requirements.txt
```

The pretrained encoders are [CXR-RePaiR ViT-B/32](https://github.com/rajpurkarlab/CXR-RePaiR) and [microsoft/BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized). Obtain the original visual checkpoint, `clip-imp-pretrained_128_6_after_4.pt`, from its upstream source. CXR-BERT can be loaded using its Hugging Face model ID or an existing local copy. Retain the upstream model licenses and citations when using or redistributing those resources.

## Data configuration

CXR-OMIT is derived from credentialed PhysioNet resources. Images, report text, derived evidence tables, and patient-level predictions are not distributed in this repository. Obtain source data through the corresponding [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/), [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/), and [CXR-LT](https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/2.0.0/) access processes. CXR-OMIT data and CGQ-Net weights are distributed separately from the code. A separate PhysioNet benchmark release is planned; its project-specific access link and the checkpoint download link will be listed in the [resource table](../README.md#resources) when available. This code release consumes prepared benchmark Parquets and does not include the benchmark construction pipeline.

**The default workflow uses CXR-OMIT v1, the version used in the paper.** The paper checkpoint, bundled thresholds, and reported metrics correspond to v1. The corrected v2 data are a separate version; the current loader does not apply its additional `coverage_valid_mask_*` fields. Keep versions in separate directories and point `CXR_OMIT_ROOT` to v1 for paper reproduction. See [data versions](../README.md#data-versions).

Set these paths in your shell, or replace the corresponding entries in the YAML files:

```bash
export CXR_OMIT_ROOT=/path/to/prepared/benchmark
export CXR_IMAGE_ROOT=/path/to/images
export CXR_REPAIR_CHECKPOINT=/path/to/clip-imp-pretrained_128_6_after_4.pt
# Optional: point to a local CXR-BERT copy instead of the Hugging Face model ID.
export CXR_BERT_PATH=/path/to/BiomedVLP-CXR-BERT-specialized
```

Place the prepared benchmark files under `CXR_OMIT_ROOT` using these names:

```text
benchmark/
  train.parquet
  val.parquet
  test.parquet
```

If your authorized copies use different filenames, rename the files or edit the three Parquet paths in both `configs/train.yaml` and `configs/eval.yaml`. Renaming a file does not change its rows, labels, or patient-disjoint split assignment.

With the default `image_path_layout: split_flat`, images are resolved under `CXR_IMAGE_ROOT/{train,val,test}/` using the filename stored in `image_path`. For an existing image directory tree, set `image_path_layout: auto` in both configurations and store image paths relative to `CXR_IMAGE_ROOT` in the Parquets.

All data and pretrained-model locations are configurable through the environment variables above or the YAML files. Training and evaluation outputs are written to relative paths under `outputs/`.

Each row contains `sample_id`, `study_id`, `image_path`, `split`, `perturbed_report_text`, and `is_clean`. Supervision uses the following category-specific columns:

| Prefix | Meaning |
|---|---|
| `weak_label_` | Source-report proxy for visual support |
| `report_label_` | Coverage of the finding in the candidate report |
| `omit_label_` | Omission target |
| `omit_mask_` | Validity for omission supervision and evaluation |
| `weak_valid_mask_` | Validity for support and coverage supervision |
| `omit_pair_weight_` | Omission pair weight |
| `tail_class_weight_` | Category weight |

The fixed category order is:

```text
atelectasis, cardiomegaly, consolidation, fracture,
pulmonary_nodule_mass, pleural_effusion, pneumothorax,
emphysema, edema
```

Masks supervise training and evaluation; they are not model inputs. The loaders retain the stored benchmark rows, including training control duplication.

## Training

Run from the repository root:

```bash
python train.py --config configs/train.yaml
```

The paper configuration uses seed 42, batch size 128, one query per category, a frozen text encoder, and fine tuning of the final visual Transformer block and positional embeddings. Queries are initialized with an independent seed-42 Gaussian generator using a `(9, 10, D)` reference tensor and retaining the first query per category. Preserve `random_reference_max_queries: 10` when reproducing the initialization.

Checkpoints are selected using validation macro-AUPRC with short-edge resizing to 544 followed by a 512 center crop. The selected paper checkpoint is epoch002 with selection macro-AUPRC 0.6333202. Resume your own run with:

```bash
python train.py --config configs/train.yaml --resume /path/to/last.ckpt
```

## Threshold fitting and evaluation

Threshold fitting and final evaluation use direct bicubic resizing to 512, distinct from the validation preprocessing used for checkpoint selection. Evaluation runs in FP32. Fit category thresholds on validation data for a newly trained checkpoint:

```bash
python optimize_thresholds.py --config configs/eval.yaml \
  --checkpoint /path/to/model.ckpt --fit-split val --eval-split test \
  --batch-size 128 --num-workers 0 \
  --output outputs/thresholds_per_class.json
```

Evaluate with frozen validation thresholds:

```bash
python evaluate.py --config configs/eval.yaml --checkpoint /path/to/model.ckpt \
  --threshold-json outputs/thresholds_per_class.json \
  --output outputs/test_metrics.json
```

`configs/thresholds_paper.json` contains thresholds for the selected paper checkpoint. Use those thresholds only with that checkpoint; refit thresholds on validation data for a new training run. `configs/paper_checkpoint.json` records the paper checkpoint identity and expected metrics. The binary checkpoint is not included in this code upload.

| Paper test metric | Value |
|---|---:|
| Macro-AUPRC | 0.6390 |
| Micro-AUPRC | 0.8034 |
| Report-AUPRC | 0.9567 |
| Macro-F1 | 0.6050 |
| Clean FAR | 0.1161 |

Category metrics use valid `omit_mask_` positions. Report risk is the maximum omission score over valid categories during benchmark evaluation. Clean FAR is the fraction of unmodified controls with any valid category crossing its validation-selected threshold.

## Inference

Provide one radiograph and a UTF-8 candidate report file:

```bash
python inference.py --config configs/eval.yaml --checkpoint /path/to/model.ckpt \
  --image /path/to/image.jpg --report-file /path/to/report.txt \
  --threshold-json configs/thresholds_paper.json --output outputs/prediction.json
```

Inference returns support, coverage, and omission probabilities, plus category-specific threshold decisions. The report risk is the maximum over all nine categories because validity masks are unavailable for a new input. Reports exceeding the 256-token paper input limit are rejected. Use `--device cpu` to run on CPU.

## Tests

The tests use synthetic inputs and do not require patient data or pretrained weights:

```bash
python -m unittest discover -s tests -v
```

They cover the visual encoder's batch independence, main-model forward and backward passes, category thresholds, validity masking, and the distinct preprocessing configurations.

## License

The code is released under the existing [MIT License](../LICENSE). Data and pretrained model licenses apply separately. Please cite the paper and the upstream resources when using this work.
