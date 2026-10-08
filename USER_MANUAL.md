# COMP700 Reproducibility User Manual

## Smart Home Appliance Identification Using Deep Learning

This manual explains how to obtain a writable project copy, open the supplied Google Colab notebook, reproduce the implemented analyses, evaluate saved models, generate thesis figures, perform LiteRT conversion and profiling, resume SHAP computations, and preserve new artefacts without mixing them with the evidence used in the thesis.

## 1. Project Contacts

| Role | Name | Institutional email |
|---|---|---|
| Project author and contact | Arjun Ramphal | [223005056@stu.ukzn.ac.za](mailto:223005056@stu.ukzn.ac.za) |
| Project author | Darian Robert | [223118685@stu.ukzn.ac.za](mailto:223118685@stu.ukzn.ac.za) |
| Supervisor | Prof. Okuthe Paul Kogeda | [KogedaO@ukzn.ac.za](mailto:KogedaO@ukzn.ac.za) |
| Supervisor | Prof. Mluleki Bryan Sinqadu | [SinqaduM@ukzn.ac.za](mailto:SinqaduM@ukzn.ac.za) |

## 2. What the Notebook Implements

The notebook implements the following research workflow:

1. mounts persistent Google Drive storage;
2. optionally obtains the public dataset;
3. audits the original publisher files;
4. quantifies publisher-boundary feature-distribution and class-support shift;
5. pools and exactly deduplicates the source files;
6. reconstructs a deterministic stratified 64/16/20 split;
7. tunes and trains five neural architectures and a Random Forest;
8. selects neural champion artefacts using validation results only;
9. evaluates locked artefacts on a 376-observation held-out set;
10. demonstrates one-flow-at-a-time inference;
11. converts selected neural models to validated FP32 LiteRT;
12. profiles prepared-input inference on x86_64 and ARM64 CPUs;
13. runs Kernel SHAP and correlation-aware Partition SHAP; and
14. performs post-hoc difficult-class error analysis.

The notebook does not implement packet capture, live flow construction, online feature extraction, open-set unknown-device rejection or policy enforcement. Profiling begins with an already prepared model tensor.

## 3. Required Services and Software

For normal use you need:

- a Google account;
- sufficient Google Drive storage;
- access to [Google Colab](https://colab.research.google.com/);
- the public project folder or GitHub repository;
- Internet access when packages or the optional Kaggle copy must be downloaded; and
- a modern browser.

An ARM64 Linux host is required only if you intend to repeat the ARM64 profiling experiment.

## 4. Obtain the Project

### Option A: Public Google Drive Folder

Open:

<https://drive.google.com/drive/folders/1_4Buza9xsp1shH3ZOvl8a-PHOVBsUwmt?usp=drive_link>

Download the complete inner `Honours_Project` folder. If Google Drive divides the download into several ZIP archives, extract every archive into the same destination while preserving the original hierarchy.

Upload the resulting `Honours_Project` folder directly to:

```text
My Drive/
└── Honours_Project/
```

The required Colab path is:

```text
/content/drive/MyDrive/Honours_Project/
```

Avoid creating an additional wrapper level.

### Option B: GitHub

Clone or download:

<https://github.com/ArjunRamphal/smart-home-appliance-identification-using-deep-learning>

The repository contains the notebook, documentation and thesis material. Large experimental artefacts remain in the public Drive distribution.

## 5. Open the Notebook

The notebook is:

```text
COMP700_Project.ipynb
```

If a downloaded Drive copy appears as an unknown binary or cannot be opened in Colab, renaming it is not sufficient. Instead:

1. download the actual `.ipynb` from GitHub;
2. open Google Colab;
3. select **File → Upload notebook**;
4. upload the `.ipynb`; and
5. select **File → Save a copy in Drive**.

The recognised copy normally appears under `Colab Notebooks`.

## 6. Choose an Execution Route

Do not use **Run all** unless you deliberately want to repeat every model search, training run and SHAP computation.

### Route A: Saved-Artefact Evaluation

Use this route to:

- inspect retained results;
- evaluate saved models;
- regenerate thesis figures;
- demonstrate inference;
- convert saved Keras models;
- repeat edge profiling; or
- regenerate SHAP tables and figures from completed arrays.

The general order is:

1. Google Drive Setup;
2. relevant saved-artefact evaluator or plotting cell;
3. converter, profiler or SHAP post-processing cell as required.

Model-training cells do not need to run in the same Colab session.

### Route B: Full Reproduction

Use this route to repeat the complete experiment:

1. Google Drive Setup;
2. optional Dataset Acquisition;
3. EDA and dataset-integrity analysis;
4. model-specific hyperparameter search and five-seed training;
5. held-out evaluation;
6. selected-model and five-seed figures;
7. LiteRT conversion and validation;
8. x86_64 profiling;
9. ARM64 bundle creation and external profiling;
10. cross-platform summary;
11. SHAP analysis; and
12. difficult-class analysis.

Every new training execution should create a new timestamped run directory rather than overwriting retained evidence.

## 7. Dataset Setup

The required files are:

```text
/content/drive/MyDrive/Honours_Project/iot_device_train.csv
/content/drive/MyDrive/Honours_Project/iot_device_test.csv
```

If they are missing, run the notebook's **Dataset Acquisition and Storage** cell.

The optional Kaggle acquisition route downloads:

<https://www.kaggle.com/datasets/fanbyprinciple/iot-device-identification>

The Packt and Kaggle CSV copies were verified as identical in dimensions, schema, inferred types, missingness, row order, labels and parsed cell values.

## 8. Run the EDA and Integrity Analysis

Run the EDA section after mounting Drive and confirming that both CSV files exist.

It performs:

- schema and target-column validation;
- duplicate and missing-value analysis;
- class-support and class-proportion analysis;
- publisher-boundary univariate distribution testing;
- Benjamini-Hochberg multiple-testing correction;
- standardised effect-size analysis;
- domain-classifier shift testing;
- PCA and t-SNE diagnostic visualisation;
- pooled-corpus deduplication;
- reconstructed-split verification;
- class-coverage checks; and
- duplicate-overlap checks.

Corrected outputs are written under:

```text
Honours_Project/EDA_Analysis_Results/
```

Publisher-boundary shift outputs are written under:

```text
Honours_Project/Covariate_Shift_Testing/
```

The publisher files show dataset shift with both feature-distribution and class-support differences. These diagnostics do not prove pure covariate shift because they do not establish that $P(Y\mid X)$ remains unchanged.

## 9. Train the Neural Models

Each neural model has its own self-contained training block. The blocks record their runtime environment, dependencies, source code, source hash, terminal output and generated artefact inventory inside the run directory.

The supported neural groups are:

```text
Models/Tuned/CNN_BiLSTM/
Models/Tuned/Feature_Attention_MLP/
Models/Tuned/CNN_Tuned/
Models/Tuned/LSTM_Tuned/
Models/Tuned/BiLSTM_Tuned/
```

Each selected configuration is trained using seeds:

```text
42, 7, 123, 2024, 99
```

The selected artefact is chosen using restored validation accuracy. Restored validation loss is used only as a tie-breaker. Held-out test results do not participate in selection.

Training new models creates new timestamped directories. Do not replace or merge existing retained directories.

## 10. Train the Random Forest

The tuned Random Forest uses:

- a preprocessing and SMOTE-aware `imblearn` pipeline;
- 30 sampled hyperparameter configurations;
- five-fold stratified cross-validation;
- training data only; and
- macro-F1 as the refit metric.

The selected configuration is:

```text
n_estimators=500
criterion=log_loss
max_depth=None
min_samples_split=10
min_samples_leaf=4
max_features=0.5
class_weight=balanced_subsample
```

Outputs are written under:

```text
Models/Baselines/Tuned_RF/<timestamp>/
```

The current Random Forest block records CPU, memory allocation, operating system, Python packages and GPU availability before fitting.

## 11. Evaluate Saved Models

Run **Held-Out Test Evaluation** after training or directly against compatible archived runs.

The evaluator supports:

- validation-selected Keras models;
- seed-specific Keras models; and
- the tuned Random Forest pipeline.

For each selected model it writes:

```text
<run>/Test_Evaluation/<model_identifier>/
├── metrics.csv
├── metrics.json
├── classification_report.csv
├── classification_report.txt
├── per_class_metrics.csv
├── predicted_classes.npy
├── prediction_probabilities.npy
├── confusion_matrix.npy
└── Figures/
```

The canonical result is:

```text
<run>/Test_Evaluation/<model_identifier>/metrics.csv
```

Run-level summaries and files under `Models/Test_Evaluation_Summaries/` are convenience copies of the same evaluations. They are not independent experiments.

When all five seed models are evaluated, a `Seed_Group_<group>` directory stores the per-seed and aggregate results required by five-seed figures.

## 12. Generate the Selected-Model Predictive Comparison

Run **Selected-Model Predictive Comparison Figure** after canonical evaluations exist for:

- Attention CNN-BiLSTM;
- Feature-Attention MLP;
- CNN;
- LSTM;
- BiLSTM; and
- tuned Random Forest.

The cell searches only canonical model-specific files:

```text
<run>/Test_Evaluation/<model_identifier>/metrics.csv
```

It does not depend on:

```text
all_model_overall_metrics.csv
```

and does not require the post-hoc difficult-class section to have run.

Global and run-level summary copies are ignored, which prevents the same evaluation from appearing multiple times in the selection menu.

Before plotting, the cell verifies:

- exactly 376 held-out observations for every model;
- identical held-out labels across all six models;
- identical class order across all six models; and
- identical processed held-out tensors across the five neural models.

Outputs are written to:

```text
Honours_Project/Thesis_Figures/
├── selected_model_predictive_comparison.png
├── selected_model_predictive_comparison.pdf
├── selected_model_predictive_comparison_data.csv
└── selected_model_predictive_comparison_provenance.json
```

## 13. Generate the CNN-BiLSTM Five-Seed Figure

Run **CNN-BiLSTM Five-Seed Performance Visualisation** after:

- all five CNN-BiLSTM seed evaluations exist; and
- the tuned Random Forest canonical evaluation exists.

The graph uses the five individual CNN-BiLSTM seed results. It marks the validation-selected seed, reports the five-seed mean and sample-standard-deviation interval, and includes the Random Forest accuracy as a comparison line.

Outputs are written under:

```text
Honours_Project/CNN_BiLSTM_Five_Seed_Performance/
```

## 14. Run the Inference Demonstration

Run **Full Test-Set Inference Demonstration** to demonstrate real predictions from a saved model without retraining.

The cell discovers compatible saved neural models and the tuned Random Forest. It then:

1. asks which model to use;
2. loads the appropriate raw or processed held-out representation;
3. predicts one observation at a time;
4. displays the predicted class, true class, confidence and correctness;
5. updates running accuracy; and
6. displays a final prediction table and running-accuracy graph.

This is a demonstration rather than the source of authoritative final metrics. The universal held-out evaluator remains the canonical evaluation route.

## 15. Convert Neural Models to LiteRT

Run **LiteRT Model Conversion and Validation** after the required saved Keras models exist.

The converter operates on artefacts stored in Google Drive, so it can be run in a later Colab session.

Selection commands include:

- `new`  -  models without completed validated conversion;
- `latest`  -  newest validation-selected artefact from each supported group;
- `all`  -  all validation-selected candidates;
- numbered selections such as `1,3,5-8`; and
- `path`  -  one explicit `.keras` path.

For LSTM, BiLSTM and CNN-BiLSTM models, the converter reconstructs a CPU-compatible static graph using `use_cudnn=False`, copies the trained weights and verifies the transfer before conversion.

A converted artefact is accepted only if:

- every output is finite;
- source Keras and deployment Keras probabilities meet the declared tolerances;
- predicted classes agree for all 376 observations;
- LiteRT accuracy and macro-F1 match the source model; and
- the frozen graph contains no remaining variable-read operations.

Accepted files are stored next to the source model:

```text
<model>_edge_static_batch1_fp32_frozen.tflite
<model>_conversion_validation.json
```

## 16. Profile Models on x86_64

Run **Controlled Edge-Inference Profiling** only on converter-validated LiteRT artefacts.

The fixed protocol is:

```text
LiteRT threads:              1
fresh processes per model:   3
warm-up inferences:          50
latency repetitions:         10
held-out flows per pass:     376
static batch size:           1
memory sample interval:      0.002 seconds
```

Each process is pinned to one eligible logical CPU where supported.

The principal timing boundary is:

```text
set_tensor → invoke → get_tensor
```

Per-model outputs are stored in the corresponding run's `Edge_Profiles/` directory.

Combined summaries are written under:

```text
Models/Edge_Profiling_Summaries/
```

## 17. Run ARM64 Profiling

### Required Script

The bundle builder expects:

```text
/content/drive/MyDrive/Honours_Project/ARM_Profiling/
    comp700_arm_selected_models_v7.py
```

Use the maintained project copy. Do not replace it with an older patched version.

### Build the Bundle in Colab

1. Ensure the required neural models have passed LiteRT conversion validation.
2. Run **ARM64 Profiling Bundle Assembly**.
3. Select the intended validation-selected neural artefacts.
4. Confirm the printed model list.
5. Type `BUILD` when prompted.
6. Obtain the generated ZIP from `ARM_Profiling_Bundles/<timestamp>/`.

### Run the Bundle

Transfer the ZIP to the ARM64 Linux system, extract it and enter the generated bundle directory.

Run:

```bash
python3 comp700_arm_selected_models_v7.py profile --auto-install --require-arm64
```

The profiling command creates a private virtual environment when needed, installs `numpy`, `psutil` and `ai-edge-litert`, and validates the bundle before executing the profile.

Preserve the complete output directory:

```text
results/
├── arm_edge_profile_summary.csv
├── arm_edge_profile_all.json
├── environment.json
└── raw/
```

### Archive the Results

Read `captured_at` from:

```text
results/environment.json
```

Create a matching timestamped directory and upload the unchanged output under:

```text
Models/Edge_Profiling_Summaries/ARM64/
└── YYYY-MM-DD_HH-MM-SS/
    └── results/
```

The ARM64 evidence was produced on an Oracle Cloud ARM64 VM.

## 18. Combine x86_64 and ARM64 Results

Run **Cross-Platform LiteRT Edge-Profiling Summary**.

The cell requires exactly one x86_64 and one ARM64 record for each neural model:

- CNN;
- Feature-Attention MLP;
- LSTM;
- BiLSTM; and
- CNN-BiLSTM.

It validates:

- platform architecture;
- model set;
- repetition count;
- thread count; and
- finite measurements.

Outputs include:

```text
Models/Edge_Profiling_Summaries/
├── Cross_Platform_Edge_Profile.png
├── Cross_Platform_Edge_Profile.pdf
├── Cross_Platform_Edge_Profile_Data.csv
└── Cross_Platform_Edge_Profile_Provenance.json
```

The provenance JSON identifies the exact source summaries.

## 19. Generate the Predictive-Deployment Trade-Off Figure

Run this figure only after:

1. canonical held-out evaluations exist;
2. selected neural models have valid LiteRT profiles; and
3. the cross-platform summary exists.

The figure includes the five neural models. Random Forest is excluded because it does not have a corresponding neural LiteRT deployment artefact.

For each platform:

- x-position is median prepared-input latency on a logarithmic scale;
- y-position is validation-selected held-out accuracy; and
- marker area represents FP32 LiteRT file size.

The generated CSV and provenance JSON record the exact evaluation and profile inputs used.

## 20. Reproduce SHAP Analysis

The authoritative SHAP directory is:

```text
Models/Tuned/CNN_BiLSTM/2026-09-27_18-12-25/SHAP_Final/
```

### Full Computation Order

Run:

1. **SHAP Common Setup and Integrity Checks**;
2. **Kernel SHAP Monte-Carlo Stability Analysis**;
3. **Full Held-Out Kernel SHAP Analysis**;
4. **Correlation-Aware Partition SHAP**; and
5. **Full-Test SHAP Post-Processing and Cross-Method Comparison**.

The setup verifies:

- validation-selected seed identity;
- model compatibility;
- held-out tensors and labels;
- feature and class order;
- preprocessing artefacts;
- reconstructed training data;
- real pre-SMOTE background observations; and
- model probabilities.

### Resuming Kernel SHAP

The full Kernel SHAP block is restartable. It reloads the necessary model and Drive artefacts, checks saved chunks and skips valid completed checkpoints.

After a runtime disconnection, rerun the full Kernel SHAP cell. Do not delete valid checkpoint files.

### Resuming Partition SHAP

After a fresh runtime:

1. rerun **SHAP Common Setup and Integrity Checks**;
2. rerun **Correlation-Aware Partition SHAP**.

The block loads complete validated tensors when available. Otherwise, it preserves valid checkpoint pairs and recomputes only missing or invalid chunks.

### Regenerating Results Without Recomputing SHAP

When complete Kernel and Partition arrays already exist, run:

**Full-Test SHAP Post-Processing and Cross-Method Comparison**

This block verifies:

- tensor shape `(376, 244, 10)`;
- finite values;
- feature order; and
- local additive fidelity.

It then regenerates feature, family and agreement tables and figures without rerunning either explainer.

## 21. Run Difficult-Class Error Analysis

The difficult classes are:

```text
lights
socket
water_sensor
```

The analysis writes to:

```text
Models/Tuned/CNN_BiLSTM/2026-09-27_18-12-25/
└── Post_Hoc_Difficult_Class_Error_Analysis/
```

### Selected-Model Case Audit

Examines:

- confusion directions;
- confidence;
- probability margins;
- predictive entropy; and
- true-class rank.

### Five-Seed Stability

Uses all five CNN-BiLSTM seed evaluations to determine whether difficult-class behaviour persists across independent seeded runs.

### Six-Model Comparison

Uses canonical evaluations for:

- CNN-BiLSTM;
- Feature-Attention MLP;
- CNN;
- LSTM;
- BiLSTM; and
- Random Forest.

Random Forest is included as a comparator. No neural-Random-Forest ensemble is constructed.

The combined post-hoc table is stored under:

```text
03_All_Model_Comparison/all_model_overall_metrics.csv
```

This file belongs only to the difficult-class analysis. It is not an input to the selected-model predictive comparison figure.

### Data-Space Analysis

Uses:

- original CSV files;
- the selected CNN-BiLSTM split indices;
- fitted preprocessing artefacts;
- real training observations for feature tests; and
- untouched validation observations for nearest-neighbour analysis.

The held-out set is not used to select a corrective intervention.

### Contrastive SHAP

Uses completed full-test Kernel and Partition SHAP tensors to explain observed difficult-class confusion directions.

All difficult-class results are post hoc. Any corrective model must be developed using training and validation evidence and tested on new independent data or a predeclared nested evaluation protocol.

## 22. Authoritative Retained Runs

| Model | Run directory | Selected artefact |
|---|---|---|
| Attention CNN-BiLSTM | `Models/Tuned/CNN_BiLSTM/2026-09-27_18-12-25/` | `best_validation_cnn_bilstm.keras`, seed 42 |
| Feature-Attention MLP | `Models/Tuned/Feature_Attention_MLP/2026-09-27_18-30-51/` | `best_validation_mlp.keras`, seed 123 |
| CNN | `Models/Tuned/CNN_Tuned/2026-09-27_18-36-23/` | `best_validation_cnn.keras`, seed 42 |
| LSTM | `Models/Tuned/LSTM_Tuned/2026-09-27_18-44-52/` | `best_validation_lstm.keras`, seed 42 |
| BiLSTM | `Models/Tuned/BiLSTM_Tuned/2026-09-27_19-03-46/` | `best_validation_bilstm.keras`, seed 123 |
| Random Forest | `Models/Baselines/Tuned_RF/2026-09-28_12-50-28/` | `tuned_random_forest_pipeline.joblib` |

## 23. Current Canonical Selected-Model Results

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Attention CNN-BiLSTM | 82.713% | 0.8172 | 0.8158 |
| Feature-Attention MLP | 81.915% | 0.8297 | 0.7953 |
| CNN | 81.383% | 0.8247 | 0.7941 |
| LSTM | 73.138% | 0.7242 | 0.7015 |
| BiLSTM | 73.138% | 0.7189 | 0.7048 |
| Tuned Random Forest | 83.511% | 0.8347 | 0.8244 |

## 24. Artefact and Provenance Rules

- Never combine a model from one run with preprocessing or test artefacts from another.
- Preserve complete timestamped run directories.
- Preserve `run_metadata.json`, `runtime_environment.json`, `system_diagnostics.json`, `requirements.txt`, source files, hashes and file inventories.
- Treat model-specific `Test_Evaluation/<identifier>/metrics.csv` as canonical.
- Treat global and run-level evaluation summaries as duplicate indexes, not independent results.
- Preserve complete ARM64 `results/` directories unchanged.
- Preserve complete SHAP checkpoint pairs and metadata.
- Do not bypass shape, hash, feature-order, class-order or equivalence failures.
- Do not overwrite retained evidence with a new experiment.

## 25. Common Problems

### Drive Path Not Found

Confirm that the project root is exactly:

```text
/content/drive/MyDrive/Honours_Project/
```

### Notebook Appears as a Binary File

Renaming the file does not convert it into notebook JSON. Upload the actual `.ipynb` through Colab and save a recognised copy to Drive.

### Dataset Missing

Run **Dataset Acquisition and Storage**, or place both CSV files directly in the project root.

### No Compatible Model Found

Confirm that the complete timestamped run directory is present. Neural runs require the model, fixed test arrays, class names and metadata. The Random Forest requires its pipeline and raw held-out bundle.

### Selected-Model Figure Cannot Find Results

Confirm that all six canonical files exist:

```text
<run>/Test_Evaluation/<model_identifier>/metrics.csv
```

Do not manually substitute a global summary CSV.

### Selection Menu Shows the Same Result Multiple Times

That behaviour belongs to older code that scans global, run-level and model-specific copies of one evaluation.

The current selected-model figure searches only:

```text
Test_Evaluation/<model_identifier>/metrics.csv
```

and presents one entry per genuine timestamped run.

### `SelfAttention` Cannot Be Located

Do not call `load_model` without registering the custom layer. Use the supplied evaluator, converter, inference or SHAP setup cell, which registers the compatible `SelfAttention` implementation before loading the CNN-BiLSTM.

### LiteRT Conversion Is Rejected

Read the conversion-validation output. A model that fails source/static/LiteRT equivalence must not be profiled or reported as validated.

### SHAP Variables Are Missing After Restart

- Rerun **SHAP Common Setup and Integrity Checks** for stability and Partition SHAP.
- Rerun the full Kernel SHAP block to resume its checkpoints.
- Use the final post-processing block when complete saved arrays already exist.

### Runtime Disconnects During a Long Operation

Reconnect, remount Drive and rerun the setup block documented for that stage. Files already written to Drive remain available.

Do not delete valid SHAP checkpoints merely because the in-memory runtime was lost.

### A New Neural Result Differs Slightly

GPU-backed TensorFlow training is not guaranteed to be byte-for-byte identical across independent runs. Preserve the new run as another replication and compare its five-seed distribution with the retained evidence.

## 26. Interpretation Limits

- The task is closed-set classification of ten known classes.
- The public corpus does not represent all households, devices, firmware versions or network environments.
- Publisher-boundary diagnostics establish dataset shift.
- Recurrent models traverse engineered-feature order rather than packet time.
- SHAP values are model-specific associations and not causal traffic mechanisms.
- The Oracle ARM64 profile establishes ARM execution and prepared-input performance.
- Profiling excludes capture, flow construction, feature extraction, network I/O, sustained throughput, power consumption and thermal behaviour.

## 27. Generative AI Declaration

OpenAI ChatGPT was used as a supporting tool for language refinement, selected code assistance, troubleshooting, and some plotting and diagram code. The authors reviewed and verified AI-assisted suggestions against the source code, datasets, saved experimental artefacts and cited literature.

The authors executed the reported experiments and retain responsibility for the methodology, interpretation, scientific integrity and final publication.
