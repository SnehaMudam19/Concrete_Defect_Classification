# Concrete Defect Classification

## Project Overview

This project applies deep learning and computer vision to classify visible conditions in
concrete-surface images. The implemented workflow treats the task as multi-label
classification: one image can contain more than one of six annotated classes. The
objective is to support automated, image-based preliminary inspection.

## Problem Statement

Concrete defects such as cracks, rust, spallation, efflorescence, and exposed
reinforcement can be difficult to review consistently at scale. This project investigates
CNN-based image classification to identify the visible labels associated with a concrete
image and to provide Grad-CAM visual explanations for a prediction.

## Objectives

- Explore DACL1K annotations and prepare RGB images with multi-label targets.
- Compare custom, frozen ResNet18, and fine-tuned ResNet18 models.
- Evaluate with macro-F1, micro-F1, loss, and per-class F1, including tuned thresholds.
- Demonstrate inference and Grad-CAM explanations for test or uploaded images.

## Dataset

The notebooks download the **DACL1K** dataset with
`bikit.utils.download_dataset("dacl1k")`. The dataset notebook records the dataset
information page as [dacl.ai](https://dacl.ai).

- **Purpose:** annotated concrete-surface imagery used by this building-inspection
  defect-classification workflow.
- **Samples:** 1,474 images described by `annotations_v1.csv`.
- **Image representation used by the code:** image paths point to `.jpg` files; images
  are opened with Pillow and converted to RGB.
- **Labels:** `NoDamage`, `Crack`, `Efflorescence`, `Rust`, `Spallation`, and
  `BarsExposed`.
- **Task type:** multi-label; 891 images have one label, 321 have two, 217 have three,
  42 have four, and 3 have five (mean: 1.61 labels per image).
- **Class counts:** Rust 667, Spallation 519, Efflorescence 383, Crack 381, NoDamage
  222, and BarsExposed 195.
- **Split:** 1,101 training images, 154 validation images, and 219 test images,
  using the dataset's `split_type` column.

The original dataset and its image files are **not included in this GitHub repository**.
They must be obtained separately from the original source before running the notebooks.

## Methodology

1. Download DACL1K, inspect its annotation CSV, and convert annotation strings into
   six binary label columns.
2. Load images as RGB, resize to 224 x 224, and normalize with ImageNet mean
   `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
3. Apply training augmentation: horizontal flip (`p=0.5`), rotation (10 degrees), and
   brightness/contrast jitter (`0.1` each). Evaluation uses resize, tensor conversion,
   and normalization only.
4. Use the supplied train, validation, and test split labels to train and compare a
   custom CNN, frozen ImageNet ResNet18, and partially fine-tuned ResNet18.
5. Select checkpoints with validation macro-F1, evaluate on test data, and optionally
   tune one probability threshold per class using validation F1.
6. Run sigmoid-based inference and produce a Grad-CAM overlay for the highest-scoring
   class.

## Model Architecture

### SimpleCNN baseline

- Four convolution blocks (`3->32->64->128->256`), each with a 3 x 3 convolution
  (padding 1), ReLU, and 2 x 2 max pooling.
- Adaptive average pooling to 1 x 1, flattening, dropout (`0.3`), and a linear layer
  from 256 features to six outputs.

### ResNet18 experiments

- The frozen model uses `torchvision.models.resnet18` with
  `ResNet18_Weights.IMAGENET1K_V1`; all original parameters are frozen and its final
  fully connected layer is replaced with six outputs.
- The fine-tuned model starts from the frozen checkpoint and unfreezes only `layer4`
  and the replacement fully connected head.
- Six independent logits are converted to per-class probabilities with sigmoid.

## Training

- **Loss:** `BCEWithLogitsLoss`.
- **Optimizer:** Adam.
- **Input:** 224 x 224 RGB images with ImageNet normalization.
- **Batch size:** 32.
- **DataLoader:** training batches are shuffled; validation and test batches are not.
- **SimpleCNN:** 15 epochs, learning rate `1e-3`.
- **Frozen ResNet18:** 8 epochs, learning rate `1e-3`.
- **Fine-tuned ResNet18:** 8 epochs, learning rate `1e-4`.
- **Reproducibility:** Python, NumPy, and PyTorch seeds are set to 42.
- **Checkpointing:** a last checkpoint supports resuming; a best checkpoint is saved
  when validation macro-F1 improves.
- No learning-rate scheduler, early stopping, or class weighting is implemented.
- Validation during training uses a fixed probability threshold of `0.5`.

## Results and Evaluation

The following results are printed in `4.Testing.ipynb` and come from the notebook
experiments. The initial comparison uses the fixed `0.5` threshold.

| Model | Test loss | Test macro-F1 | Test micro-F1 |
| --- | ---: | ---: | ---: |
| SimpleCNN | 0.6489 | 0.1376 | 0.3151 |
| ResNet18 frozen | 0.5464 | 0.4826 | 0.5313 |
| ResNet18 fine-tuned | 0.5352 | 0.6483 | 0.6616 |

Fine-tuned ResNet18 per-class test F1 at the fixed threshold:

| Class | F1 |
| --- | ---: |
| NoDamage | 0.7595 |
| Crack | 0.4960 |
| Efflorescence | 0.7297 |
| Rust | 0.7590 |
| Spallation | 0.6391 |
| BarsExposed | 0.5063 |

The notebook then selects class-specific thresholds from `0.05` through `0.90` on the
validation split. Tuned fine-tuned ResNet18 test macro-F1 is `0.7028` and micro-F1 is
`0.7156`; per-class F1 is NoDamage `0.7952`, Crack `0.5522`, Efflorescence `0.7799`,
Rust `0.8000`, Spallation `0.7081`, and BarsExposed `0.5814`.

## Notebook Workflow

Recommended execution order:

1. **`1.Dataset.ipynb`** - installs download helpers, downloads DACL1K, inspects
   annotations, converts labels, reports counts, and samples images.
2. **`2.Eda.ipynb`** - loads the annotations, examines label combinations and class
   distribution, and visualizes sampled images.
3. **`3.Training.ipynb`** - prepares datasets and transforms, trains SimpleCNN and
   ResNet18 variants, and records checkpoints and validation histories.
4. **`4.Testing.ipynb`** - evaluates the saved best checkpoints, compares models, and
   tunes class-specific thresholds on validation predictions before test evaluation.
5. **`5.Gradcam_inference.ipynb`** - loads the fine-tuned ResNet18, predicts test or
   uploaded images, and generates Grad-CAM heatmaps and overlays.
6. **`6.App.ipynb`** - generates a Streamlit application for image upload,
   probabilities, predicted labels, and Grad-CAM visualization.

## Project Structure

Only these project files are intended to be uploaded:

```text
.
├── 1.Dataset.ipynb
├── 2.Eda.ipynb
├── 3.Training.ipynb
├── 4.Testing.ipynb
├── 5.Gradcam_inference.ipynb
├── 6.App.ipynb
├── requirements.docx
└── README.md
```

Dataset files, generated outputs, checkpoints, prediction files, and locally generated
Streamlit application files are excluded.

## Technologies Used

- Python and Jupyter/Google Colab
- PyTorch and TorchVision
- NumPy, pandas, Pillow, scikit-learn, and Matplotlib
- Streamlit and pyngrok (used by the app notebook)
- `building-inspection-toolkit` (`bikit`) and `patool` for dataset acquisition

## Setup and Installation

The notebooks target Google Colab and Google Drive paths. A GPU is recommended; the
training notebook selects CUDA when available and otherwise uses CPU.

Install the dependencies listed in `requirements.docx`, or install the notebook
dependencies in a compatible Python environment:

```bash
pip install torch torchvision pandas numpy scikit-learn Pillow matplotlib
pip install streamlit pyngrok
pip install patool git+https://github.com/phiyodr/building-inspection-toolkit
```

Obtain DACL1K separately before running the dataset notebook; no repository-local path
or download command is assumed here.

## Running the Project

Open the notebooks in Google Colab, mount Google Drive when prompted, and run them in
the order shown in [Notebook Workflow](#notebook-workflow). Later notebooks expect
the dataset and checkpoints produced by earlier notebooks to be available.

## Key Implementation Details

- Six-output sigmoid classification supports multiple labels per image.
- Inputs use 224 x 224 RGB images, ImageNet normalization, and three augmentations.
- Comparison covers a four-block CNN and two ResNet18 configurations; fine-tuning updates
  only `layer4` and the six-output head.
- Validation macro-F1 selects checkpoints and tunes thresholds; Grad-CAM targets
  `model.layer4[-1]`.

## Limitations

- The dataset and trained checkpoints are not included; data and model artifacts must
  be supplied separately.
- Reported metrics come from stored notebook experiments and may vary when retrained.
- Labels are multi-label and class counts are imbalanced.
- Grad-CAM highlights model activations; it is not a structural-safety assessment.
- The app notebook explicitly presents the model as preliminary assistance, not a
  replacement for engineering inspection.

## Future Improvements

- Add cross-validation and repeated runs to quantify variability.
- Investigate imbalance-aware objectives, sampling, augmentation, and calibration.
- Add reproducible data/checkpoint configuration and broader per-class evaluation.

## Conclusion

This project implements an end-to-end DACL1K concrete-image workflow from annotation
exploration through model comparison, threshold tuning, testing, and Grad-CAM inference.
The fine-tuned ResNet18 has the strongest fixed-threshold test macro-F1, while
validation-based threshold tuning reports a higher macro-F1 for the same model.