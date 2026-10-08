# Smart Home Appliance Identification Using Deep Learning

Reproducible COMP700 Honours project for identifying ten known smart-home IoT appliance classes from payload-independent engineered network-traffic features.

The project audits and reconstructs the publisher-defined dataset split, tunes and compares five neural architectures and a Random Forest, evaluates validation-selected artefacts on one fixed held-out set, interprets the selected CNN-BiLSTM using SHAP, converts the neural models to FP32 LiteRT, and profiles prepared-input inference on x86_64 and ARM64 CPUs.

## Project Links

- Source repository: <https://github.com/ArjunRamphal/smart-home-appliance-identification-using-deep-learning>
- Downloadable project folder: <https://drive.google.com/drive/folders/1_4Buza9xsp1shH3ZOvl8a-PHOVBsUwmt?usp=drive_link>
- Packt dataset source: <https://github.com/PacktPublishing/Machine-Learning-for-Cybersecurity-Cookbook/tree/master/Chapter05/IoT%20Device%20Type%20Identification%20Using%20Machine%20Learning>
- Equivalent Kaggle copy: <https://www.kaggle.com/datasets/fanbyprinciple/iot-device-identification>

The public Google Drive folder is viewer-only. Download the complete inner `Honours_Project` folder and upload it directly to the root of your own Google Drive before running notebook cells that create or modify artefacts.

## Implemented Scope

The implemented workflow includes:

- source-file integrity, duplicate, missingness, class-support and distribution analysis;
- pooling and exact deduplication before splitting;
- a fixed stratified 64/16/20 training, validation and test reconstruction;
- training-only variance filtering, imputation and scaling;
- training-only SMOTE;
- explicit neural hyperparameter searches and five-seed training;
- training-fold-only Random Forest search;
- validation-governed neural champion selection;
- saved-artefact held-out evaluation;
- one-flow-at-a-time inference demonstration;
- FP32 static-batch-one LiteRT conversion and numerical-equivalence validation;
- controlled prepared-input profiling on x86_64 and ARM64;
- Kernel SHAP stability analysis, full-test Kernel SHAP and correlation-aware Partition SHAP; and
- post-hoc diagnosis of the difficult `lights`, `socket` and `water_sensor` classes.

The project does not implement packet capture, live flow reconstruction, online feature extraction, unknown-device rejection, policy enforcement or a complete smart-home gateway application. Reported latency measures prepared-input model inference rather than end-to-end traffic processing.

## Dataset and Reconstructed Split

The publisher supplies 1,000 training rows and 900 testing rows, with 297 predictors and the target column `device_category`.

The original testing file omits `water_sensor`. At the original publisher boundary:

- 111 of 160 active numeric predictors remain significantly different after Benjamini-Hochberg correction; and
- a domain classifier distinguishes file membership with repeated 5-by-5 ROC-AUC of `0.999962 ± 0.000055`.

These results demonstrate feature-distribution and class-support shift. They are not presented as proof of pure covariate shift because stability of $P(Y\mid X)$ was not established.

The corrected modelling pipeline:

1. pools both publisher files;
2. removes 23 exact duplicate rows;
3. constructs a fixed stratified split with `random_state=42`;
4. learns all preprocessing from the training partition only; and
5. applies SMOTE only to processed training observations.

| Partition | Observations |
|---|---:|
| Training | 1,200 |
| Validation | 301 |
| Held-out test | 376 |

Training-derived variance filtering removes 53 constant predictors and retains 244 predictors. There is no exact feature-row overlap between reconstructed partitions.

## Models

All neural searches optimise validation accuracy. Each selected neural configuration is trained using seeds `42`, `7`, `123`, `2024` and `99`.

The representative artefact for each neural architecture is chosen using the highest restored validation accuracy, with lower restored validation loss used only as a tie-breaker. Held-out test metrics do not participate in hyperparameter or seed selection.

| Model | Selected configuration |
|---|---|
| Attention CNN-BiLSTM | 64 Conv1D filters; 64 BiLSTM units per direction; dropout 0.3; Adam learning rate `5e-4` |
| Feature-Attention MLP | hidden widths 256 and 256; fixed 64-unit third layer; dropout 0.4; Adam learning rate `1e-3` |
| CNN | 32 filters; kernel size 3; 64 dense units; dropout 0.3; Adam learning rate `1e-3` |
| LSTM | 64 recurrent units; 64 dense units; dropout 0.2; Adam learning rate `5e-4` |
| BiLSTM | 32 recurrent units per direction; 64 dense units; dropout 0.2; Adam learning rate `1e-3` |
| Tuned Random Forest | 500 trees; `log_loss`; unlimited depth; `min_samples_split=10`; `min_samples_leaf=4`; `max_features=0.5`; `balanced_subsample` |

The neural input shape `(244, 1)` represents a fixed engineered-feature ordering, not packet time. The recurrent models therefore traverse feature order and must not be described as modelling a raw temporal packet sequence.

## Authoritative Predictive Results

| Model | Five-seed held-out accuracy, mean ± sample SD | Selected-model accuracy | Selected macro-F1 |
|---|---:|---:|---:|
| Attention CNN-BiLSTM | 81.543% ± 1.227 pp | 82.713% | 0.8158 |
| Feature-Attention MLP | 81.117% ± 0.921 pp | 81.915% | 0.7953 |
| CNN | 80.798% ± 1.856 pp | 81.383% | 0.7941 |
| LSTM | 66.596% ± 6.101 pp | 73.138% | 0.7015 |
| BiLSTM | 69.947% ± 4.438 pp | 73.138% | 0.7048 |
| Tuned Random Forest | Not a five-seed neural experiment | 83.511% | 0.8244 |

The selected CNN-BiLSTM is the strongest selected neural model. The tuned Random Forest has the strongest single selected-model predictive result.

## Prepared-Input LiteRT Profiling

The five selected neural models were converted to static-batch-one FP32 LiteRT artefacts and validated against their Keras sources before profiling.

| Model | LiteRT size (KiB) | x86_64 median (ms) | ARM64 median (ms) | x86_64 peak USS overhead (MiB) | ARM64 peak USS overhead (MiB) |
|---|---:|---:|---:|---:|---:|
| CNN | 974.2 | 0.0413 | 0.0437 | 14.167 | 11.544 |
| Feature-Attention MLP | 805.8 | 0.0247 | 0.0399 | 13.622 | 11.116 |
| LSTM | 95.7 | 0.9535 | 1.1151 | 12.353 | 9.876 |
| BiLSTM | 77.2 | 1.2603 | 1.2288 | 12.409 | 9.971 |
| Attention CNN-BiLSTM | 333.3 | 3.8484 | 3.9788 | 13.775 | 11.232 |

Profiling uses:

- one inference thread;
- one pinned logical CPU;
- three fresh processes per model;
- 50 warm-up inferences per phase;
- 10 passes over all 376 held-out observations per process; and
- the timed boundary `set_tensor → invoke → get_tensor`.

The ARM64 environment is an Oracle Cloud Ubuntu 24.04.5 VM with four Arm Neoverse-N1 cores, approximately 23.4 GiB RAM, Python 3.12.3 and `ai-edge-litert` 2.2.0. It is an actual ARM64 environment, not a Raspberry Pi simulation.

## Explainability and Difficult-Class Findings

The validation-selected seed-42 CNN-BiLSTM is interpreted after model selection.

- Three independent 4,096-sample Kernel SHAP estimates on the same 100-case audit subset produced all-feature Spearman correlations from 0.9264 to 0.9375.
- Full-test Kernel and Partition SHAP over all 376 held-out cases produced all-feature Spearman correlation 0.8230.
- The two methods produced 60% top-10, 50% top-20 and 66% top-50 overlap.
- Packet-size and byte-volume, timing and inter-arrival, and TTL were the three leading signal families under both methods.
- `lights`, `socket` and `water_sensor` comprise 26.6% of held-out observations but 73.8% of selected-model errors.
- Collapsing these three labels into one difficult group produces 99.73% group-membership accuracy, showing that most errors occur within the group rather than between the group and the other classes.

The difficult-class investigation is post hoc. It explains the locked models but cannot be used to select an improved model using the same held-out set.

## Repository Contents

```text
.
├── COMP700_Project.ipynb
├── iot_device_train.csv
├── iot_device_test.csv
├── README.md
├── USER_MANUAL.md
├── third_party/
│   └── packt-dataset/
│       └── LICENSE.txt
└── thesis/
    └── COMP700_Thesis.pdf
```

Large models, fitted preprocessors, fixed test tensors, SHAP arrays, conversion artefacts and raw profiling outputs are distributed through the public Google Drive folder rather than ordinary Git history.

## Required Google Drive Structure

```text
Honours_Project/
├── COMP700 Project
├── iot_device_train.csv
├── iot_device_test.csv
├── ARM_Profiling/
│   └── comp700_arm_selected_models_v7.py
├── ARM_Profiling_Bundles/
├── CNN_BiLSTM_Five_Seed_Performance/
├── Covariate_Shift_Testing/
├── EDA_Analysis_Results/
├── Thesis_Figures/
└── Models/
    ├── Tuned/
    │   ├── CNN_BiLSTM/
    │   ├── Feature_Attention_MLP/
    │   ├── CNN_Tuned/
    │   ├── LSTM_Tuned/
    │   └── BiLSTM_Tuned/
    ├── Baselines/
    │   └── Tuned_RF/
    ├── Edge_Profiling_Summaries/
    └── Test_Evaluation_Summaries/
```

The notebook expects:

```text
/content/drive/MyDrive/Honours_Project/
```

Do not introduce an extra wrapper such as:

```text
Honours_Project_Public/Honours_Project/
```

or:

```text
Honours_Project/Honours_Project/
```

## Quick Start

1. Download the complete inner `Honours_Project` folder from the public Drive link.
2. Extract all ZIP archives while preserving their folder hierarchy.
3. Upload `Honours_Project` directly to the root of your own **My Drive**.
4. Open `COMP700_Project.ipynb` in Google Colab.
5. Run **Google Drive Setup**.
6. Read the notebook's **How to Run** guide.
7. Follow either the saved-artefact route or full-reproduction route described in `USER_MANUAL.md`.

Do not use **Run all** unless you intentionally want to retrain every model and recompute the complete SHAP workflow.

If Drive displays the notebook as an unopenable binary file, do not merely rename it. Download the actual `.ipynb` from GitHub, upload it through Colab's notebook uploader and save a recognised copy to Drive.

## Notebook Map

| Stage | Notebook section | Purpose |
|---|---|---|
| 1 | Environment and Data Setup | Mount Drive and optionally obtain missing CSV files |
| 2 | Exploratory Data Analysis and Dataset Integrity | Audit publisher files and validate the reconstructed split |
| 3 | Model Development | Tune and train the CNN-BiLSTM, MLP, CNN, LSTM, BiLSTM and Random Forest |
| 4 | Model Evaluation and Deployment | Evaluate saved artefacts, generate figures, demonstrate inference, convert models and profile x86_64 inference |
| 5 | ARM64 Profiling | Build a validated bundle, profile on ARM64 and archive unchanged outputs |
| 6 | Explainable AI Analysis | Run Kernel SHAP stability, full-test Kernel SHAP, Partition SHAP and saved-result post-processing |
| 7 | Post-hoc Difficult-Class Error Analysis | Diagnose difficult-class errors across seeds, models, feature space and SHAP explanations |

## Canonical Artefact Map

| Evidence | Canonical location |
|---|---|
| Selected model metrics | `<run>/Test_Evaluation/<model_identifier>/metrics.csv` |
| Five-seed metrics | `<run>/Test_Evaluation/Seed_Group_<group>/per_seed_test_metrics.csv` |
| Post-hoc six-model comparison | `Post_Hoc_Difficult_Class_Error_Analysis/03_All_Model_Comparison/all_model_overall_metrics.csv` |
| x86_64 profile | `Models/Edge_Profiling_Summaries/edge_profile_summary_2026-09-27_20-23-59.csv` |
| ARM64 profile | `Models/Edge_Profiling_Summaries/ARM64/2026-09-27_18-35-58/results/` |
| Cross-platform profile | `Models/Edge_Profiling_Summaries/Cross_Platform_Edge_Profile_Data.csv` |
| Final SHAP analysis | `Models/Tuned/CNN_BiLSTM/2026-09-27_18-12-25/SHAP_Final/` |
| Difficult-class analysis | `Models/Tuned/CNN_BiLSTM/2026-09-27_18-12-25/Post_Hoc_Difficult_Class_Error_Analysis/` |

`Models/Test_Evaluation_Summaries/` contains convenience copies and not independent experiments.

The selected-model comparison searches only canonical model-specific files under:

```text
<run>/Test_Evaluation/<model_identifier>/metrics.csv
```

It does not require `all_model_overall_metrics.csv`.

## Reproducibility Rules

- Preserve the fixed split, feature order, class order and declared seeds.
- Fit filtering, imputation, scaling and resampling from training data only.
- Apply SMOTE only to processed training observations.
- Select neural champions from validation evidence only.
- Select the Random Forest through training-fold cross-validation only.
- Keep the 376-observation held-out set unchanged.
- Preserve complete timestamped run directories.
- Do not combine artefacts from different runs.
- Do not bypass hash, shape, class-order, feature-order or numerical-equivalence failures.

## Data Source and Redistribution

The CSV files are sourced from Chapter 5 of Packt Publishing's *Machine Learning for Cybersecurity Cookbook* repository. The Packt and Kaggle copies were verified as identical in dimensions, schema, inferred types, missingness, row order, labels and parsed cell values.

The Packt repository distributes its Chapter 5 materials under the MIT License. Keep `third_party/packt-dataset/LICENSE.txt` with redistributed copies.

## Limitations

- The task is closed-set classification of ten known classes.
- Unknown-device detection is not implemented.
- The dataset contains engineered flow features rather than raw packets.
- The recurrent-model input order is not a genuine temporal sequence.
- The held-out set comes from the same pooled public corpus rather than an independent household capture.
- Post-hoc and SHAP analyses cannot justify test-guided retraining.
- LiteRT profiles exclude packet capture, feature extraction, network I/O, power, thermal behaviour and sustained throughput.
- The Oracle ARM64 VM establishes architecture portability but is not a Raspberry Pi measurement.

## Generative AI Disclosure

OpenAI ChatGPT was used as a supporting tool for language refinement, selected code assistance, troubleshooting, and some plotting and diagram code. The authors reviewed and verified AI-assisted suggestions against the source code, datasets, saved artefacts and cited literature. The authors executed the reported experiments and retain responsibility for the methodology, interpretation, scientific integrity and final publication.

## Project Team

- **Arjun Ramphal**  -  project author and contact  -  [223005056@stu.ukzn.ac.za](mailto:223005056@stu.ukzn.ac.za)
- **Darian Robert**  -  project author  -  [223118685@stu.ukzn.ac.za](mailto:223118685@stu.ukzn.ac.za)
- **Prof. Okuthe Paul Kogeda**  -  supervisor  -  [KogedaO@ukzn.ac.za](mailto:KogedaO@ukzn.ac.za)
- **Prof. Mluleki Bryan Sinqadu**  -  supervisor  -  [SinqaduM@ukzn.ac.za](mailto:SinqaduM@ukzn.ac.za)

School of Mathematics, Statistics and Computer Science, University of KwaZulu-Natal, South Africa.
