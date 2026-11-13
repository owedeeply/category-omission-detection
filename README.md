<div align="center">

# CGQ-Net

**Category-Guided Query Network**  
Category-level omission detection in chest X-ray reports

[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg)](docs/USAGE.md#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](requirements.txt)

[Resources](#resources) · [Quick start](#quick-start) · [Main results](#main-results) · [Full documentation](docs/USAGE.md)

</div>

Official implementation of **CGQ-Net (Category-Guided Query Network)** for category-level omission detection in chest X-ray reports. CGQ-Net performs category-specific visual and textual evidence retrieval and coverage-gated support–coverage reasoning, trained with controlled counterfactual supervision.

## Framework

[![CGQ-Net architecture: category-guided visual and textual evidence retrieval, coverage-gated reasoning, and category-level omission prediction.](assets/cgq_net_architecture.png)](assets/cgq_net_architecture.pdf)

*CGQ-Net retrieves visual and textual evidence with learnable category queries, estimates visual support and report coverage, and uses coverage-gated evidence fusion to predict omissions across nine finding categories.* [View the original vector figure](assets/cgq_net_architecture.pdf).

## Resources

Code, benchmark data, and model weights are distributed separately. The repository contains the main-model implementation, configurations, validation-selected thresholds, checkpoint metadata, and synthetic-input tests.

| Resource | Location and status |
|---|---|
| Main-model code | Available in this repository; see [full usage instructions](docs/USAGE.md). |
| CXR-OMIT benchmark | Distributed separately from this code repository. A separate PhysioNet release is planned; the project-specific access link will be added here. |
| CGQ-Net paper checkpoint | Distributed separately from the code; the download link will be added here. [Checkpoint identity, SHA-256, and expected metrics](configs/paper_checkpoint.json) are available. |
| Paper thresholds | Available in [configs/thresholds_paper.json](configs/thresholds_paper.json); use with the corresponding paper checkpoint. |
| Visual encoder initialization | [CXR-RePaiR ViT-B/32](https://github.com/rajpurkarlab/CXR-RePaiR); obtain `clip-imp-pretrained_128_6_after_4.pt` from the upstream project for training. |
| Text encoder and tokenizer | [BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized); use the model ID or a local copy. |
| Paper | The manuscript link and citation will be added when publicly available. |

Source images, report text, and benchmark Parquets are not hosted on GitHub. Access to the underlying [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/), [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/), and [CXR-LT](https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/2.0.0/) resources follows their respective access requirements. The planned benchmark release will provide its own access instructions.

## Quick start

The evaluation example below requires an authorized copy of **CXR-OMIT** and the matching CGQ-Net paper checkpoint. Their project-specific download links are not yet listed above. The repository consumes prepared benchmark files; it does not contain the benchmark construction pipeline.

### 1. Install

Use Python 3.10 or 3.11 with a PyTorch build appropriate for your hardware. Run commands from the repository root.

```bash
git clone https://github.com/owedeeply/category-omission-detection.git
cd category-omission-detection
python -m pip install -r requirements.txt
```

### 2. Configure the data

The following shell examples use Linux/macOS syntax. Set paths to your authorized local copies:

```bash
export CXR_OMIT_ROOT=/path/to/cxr_omit
export CXR_IMAGE_ROOT=/path/to/images
# Optional: use a local CXR-BERT copy instead of its Hugging Face model ID.
export CXR_BERT_PATH=/path/to/BiomedVLP-CXR-BERT-specialized
```

Leave `CXR_BERT_PATH` unset to use `microsoft/BiomedVLP-CXR-BERT-specialized` directly. Place the prepared benchmark files under `CXR_OMIT_ROOT`:

```text
cxr_omit/
├── train.parquet
├── val.parquet
└── test.parquet
```

By default, images are resolved as `CXR_IMAGE_ROOT/{train,val,test}/<image filename>`. For an existing directory tree with image paths relative to `CXR_IMAGE_ROOT`, set `data.image_path_layout: auto` in both YAML configurations. See [data configuration](docs/USAGE.md#data-configuration) for the schema and path options.

### 3. Evaluate the paper checkpoint

After obtaining the matching checkpoint, evaluate the test split with the frozen validation-selected thresholds:

```bash
python evaluate.py \
  --config configs/eval.yaml \
  --checkpoint /path/to/cgq_net_seed42_epoch002.ckpt \
  --threshold-json configs/thresholds_paper.json \
  --split test \
  --output outputs/test_metrics.json
```

The command writes metrics to `outputs/test_metrics.json`. The full trained checkpoint contains the visual encoder state, so this evaluation command does not need a separate CXR-RePaiR initialization checkpoint. CXR-BERT model/tokenizer resources are still required.

### 4. Train or run inference

For training, also configure the upstream visual initialization checkpoint:

```bash
export CXR_REPAIR_CHECKPOINT=/path/to/clip-imp-pretrained_128_6_after_4.pt
python train.py --config configs/train.yaml
```

For a newly trained checkpoint, fit thresholds on the validation split before test evaluation. See [threshold fitting](docs/USAGE.md#threshold-fitting-and-evaluation) and [single-image/report inference](docs/USAGE.md#inference) for the complete commands.

## Main results

Comparison with representative baselines on the **CXR-OMIT test split**. Finding-specific columns report per-category AUPRC. **Head**, **Middle**, and **Tail** denote frequency groups defined by the number of unique omission-positive source reports in the training set. The best result in each column is shown in **bold**.

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="3" align="center">Head</th>
      <th colspan="3" align="center">Middle</th>
      <th colspan="3" align="center">Tail</th>
      <th colspan="5" align="center">Overall Metrics</th>
    </tr>
    <tr>
      <th align="center">Atel.</th>
      <th align="center">PE</th>
      <th align="center">Cardio.</th>
      <th align="center">Edema</th>
      <th align="center">Cons.</th>
      <th align="center">PTX</th>
      <th align="center">Frac.</th>
      <th align="center">PNM</th>
      <th align="center">Emphy.</th>
      <th align="center">Macro-AUPRC<br>↑</th>
      <th align="center">Micro-AUPRC<br>↑</th>
      <th align="center">Report-AUPRC<br>↑</th>
      <th align="center">Macro-F1<br>↑</th>
      <th align="center">Clean FAR<br>↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Text-only</td>
      <td align="right">0.7079</td>
      <td align="right">0.7646</td>
      <td align="right">0.6024</td>
      <td align="right">0.7574</td>
      <td align="right">0.4480</td>
      <td align="right">0.3638</td>
      <td align="right">0.0845</td>
      <td align="right">0.1183</td>
      <td align="right">0.1075</td>
      <td align="right">0.4394</td>
      <td align="right">0.6561</td>
      <td align="right">0.9096</td>
      <td align="right">0.4644</td>
      <td align="right">0.2266</td>
    </tr>
    <tr>
      <td>Image-only</td>
      <td align="right">0.4058</td>
      <td align="right">0.3793</td>
      <td align="right">0.2516</td>
      <td align="right">0.4903</td>
      <td align="right">0.2647</td>
      <td align="right">0.1795</td>
      <td align="right">0.0690</td>
      <td align="right">0.1001</td>
      <td align="right">0.1617</td>
      <td align="right">0.2558</td>
      <td align="right">0.3332</td>
      <td align="right">0.5874</td>
      <td align="right">0.3337</td>
      <td align="right">0.7996</td>
    </tr>
    <tr>
      <td>Concat MLP</td>
      <td align="right">0.7804</td>
      <td align="right">0.8677</td>
      <td align="right">0.7407</td>
      <td align="right">0.8637</td>
      <td align="right">0.5787</td>
      <td align="right">0.4977</td>
      <td align="right">0.1187</td>
      <td align="right">0.1964</td>
      <td align="right">0.3198</td>
      <td align="right">0.5515</td>
      <td align="right">0.7519</td>
      <td align="right">0.9354</td>
      <td align="right">0.5386</td>
      <td align="right">0.1884</td>
    </tr>
    <tr>
      <td>BioViL-Sup</td>
      <td align="right">0.7815</td>
      <td align="right">0.8178</td>
      <td align="right">0.7170</td>
      <td align="right">0.8286</td>
      <td align="right">0.5756</td>
      <td align="right">0.5173</td>
      <td align="right">0.1192</td>
      <td align="right">0.1463</td>
      <td align="right">0.1889</td>
      <td align="right">0.5214</td>
      <td align="right">0.7253</td>
      <td align="right">0.9324</td>
      <td align="right">0.5169</td>
      <td align="right">0.1745</td>
    </tr>
    <tr>
      <td>Standard Cross-Attention</td>
      <td align="right">0.8324</td>
      <td align="right">0.9032</td>
      <td align="right">0.8147</td>
      <td align="right">0.8849</td>
      <td align="right">0.6481</td>
      <td align="right">0.6493</td>
      <td align="right">0.1348</td>
      <td align="right">0.1895</td>
      <td align="right">0.3099</td>
      <td align="right">0.5963</td>
      <td align="right">0.7896</td>
      <td align="right">0.9509</td>
      <td align="right">0.5717</td>
      <td align="right">0.1344</td>
    </tr>
    <tr>
      <td>Decoupled Support–Coverage</td>
      <td align="right">0.6550</td>
      <td align="right">0.7795</td>
      <td align="right">0.5250</td>
      <td align="right">0.7697</td>
      <td align="right">0.4000</td>
      <td align="right">0.3414</td>
      <td align="right">0.0803</td>
      <td align="right">0.1098</td>
      <td align="right">0.1962</td>
      <td align="right">0.4286</td>
      <td align="right">0.6086</td>
      <td align="right">0.8359</td>
      <td align="right">0.4413</td>
      <td align="right">0.4837</td>
    </tr>
    <tr>
      <td>EVA-X + Rule Coverage</td>
      <td align="right">0.6167</td>
      <td align="right">0.5741</td>
      <td align="right">0.3750</td>
      <td align="right">0.7255</td>
      <td align="right">0.4583</td>
      <td align="right">0.4832</td>
      <td align="right">0.2254</td>
      <td align="right">0.2162</td>
      <td align="right">0.2724</td>
      <td align="right">0.4385</td>
      <td align="right">0.5225</td>
      <td align="right">0.7157</td>
      <td align="right">0.4777</td>
      <td align="right">0.5801</td>
    </tr>
    <tr>
      <td>EVA-X + Learned Coverage</td>
      <td align="right">0.7050</td>
      <td align="right">0.8336</td>
      <td align="right">0.5872</td>
      <td align="right">0.7974</td>
      <td align="right">0.4604</td>
      <td align="right">0.4911</td>
      <td align="right">0.2845</td>
      <td align="right">0.2480</td>
      <td align="right">0.3076</td>
      <td align="right">0.5239</td>
      <td align="right">0.6642</td>
      <td align="right">0.8684</td>
      <td align="right">0.5132</td>
      <td align="right">0.3639</td>
    </tr>
    <tr>
      <td>KAD-ZS + Learned Coverage</td>
      <td align="right">0.6997</td>
      <td align="right">0.8453</td>
      <td align="right">0.6011</td>
      <td align="right">0.7963</td>
      <td align="right">0.4662</td>
      <td align="right">0.5208</td>
      <td align="right"><strong>0.2893</strong></td>
      <td align="right">0.2331</td>
      <td align="right">0.1710</td>
      <td align="right">0.5136</td>
      <td align="right">0.6524</td>
      <td align="right">0.8606</td>
      <td align="right">0.5071</td>
      <td align="right">0.3328</td>
    </tr>
    <tr>
      <td><strong>CGQ-Net (Ours)</strong></td>
      <td align="right"><strong>0.8629</strong></td>
      <td align="right"><strong>0.9116</strong></td>
      <td align="right"><strong>0.8292</strong></td>
      <td align="right"><strong>0.9121</strong></td>
      <td align="right"><strong>0.7303</strong></td>
      <td align="right"><strong>0.7182</strong></td>
      <td align="right">0.1994</td>
      <td align="right"><strong>0.2556</strong></td>
      <td align="right"><strong>0.3319</strong></td>
      <td align="right"><strong>0.6390</strong></td>
      <td align="right"><strong>0.8034</strong></td>
      <td align="right"><strong>0.9567</strong></td>
      <td align="right"><strong>0.6050</strong></td>
      <td align="right"><strong>0.1161</strong></td>
    </tr>
  </tbody>
</table>

Atel. = atelectasis, PE = pleural effusion, Cardio. = cardiomegaly, Cons. = consolidation, PTX = pneumothorax, Frac. = fracture, PNM = pulmonary nodule/mass, and Emphy. = emphysema.

CGQ-Net results use the selected paper checkpoint with training seed **42**, batch size **128**, and zero-based epoch **2**. Macro-F1 and Clean FAR use category-specific thresholds selected on the validation set.

Category metrics use valid omission-mask positions. Report-AUPRC uses the maximum score over valid categories; Clean FAR is the fraction of unmodified controls with any valid category crossing its validation-selected threshold. Checkpoint selection and final evaluation use different preprocessing configurations, documented in [USAGE.md](docs/USAGE.md#threshold-fitting-and-evaluation).

Exact metric values and the checkpoint checksum are recorded in [configs/paper_checkpoint.json](configs/paper_checkpoint.json). The bundled thresholds belong to that checkpoint; refit them on validation data for a new training run.

## Documentation and tests

- [Installation, data schema, training, threshold fitting, inference, and evaluation](docs/USAGE.md)
- [Paper training configuration](configs/train.yaml) and [final evaluation configuration](configs/eval.yaml)
- Synthetic tests that require no patient data or pretrained weights:

```bash
python -m unittest discover -s tests -v
```

## Citation

Citation information will be provided when the manuscript becomes publicly available.

## License

The code is released under the [MIT License](LICENSE). Data and pretrained model licenses apply separately.
