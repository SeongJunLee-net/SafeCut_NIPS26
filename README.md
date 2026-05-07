# Safeguarding Mutual Correction in Source-Free Domain Adaptation via Cut Statistics

This repository contains the official, anonymized PyTorch implementation for our submission. The code has been specifically prepared for the review process, ensuring all identifying information and previous naming conventions have been removed.

## 📁 Project Structure

```text
SafeCut_Submission_18248/
├── run_all.sh               # Master script to run all four datasets sequentially
├── run_office_home.sh       # Execution script for Office-Home dataset
├── run_office31.sh          # Execution script for Office-31 dataset
├── run_domainnet126.sh      # Execution script for DomainNet-126 dataset
├── run_visda_c.sh           # Execution script for VisDA-C dataset
├── src/
│   └── methods/
│       └── safecut/         # Core SafeCut algorithm implementation
│           ├── train.py     # Main entry point for training
│           ├── trainer.py   # Training loop and logic (Target & Peer Models)
│           ├── cut_stat.py  # Cut statistics calculation logic
│           └── prompt.py    # CLIP Prompt Learner implementation
├── clip/                    # Customized CLIP model implementation
├── data/                    # Dataset configuration and data loading scripts
├── pretrained_model/        # Directory for pre-trained source models (Backbones)
├── outer_log/               # Directory where execution logs are saved automatically
└── result/                  # Directory where result CSVs and Checkpoints are stored
```

## Execution Guide

We provide simple shell scripts to easily run the SafeCut adaptation process across different benchmark datasets. 

### 1. Running a Single Dataset
You can run the adaptation on a specific dataset using its dedicated execution script. The scripts take the `GPU_ID` as an optional argument (defaults to `0`).

```bash
# Run on Office-Home using GPU 0
bash run_office_home.sh 0

# Run on Office-31 using GPU 1
bash run_office31.sh 1

# Run on DomainNet-126 using GPU 0
bash run_domainnet126.sh 0

# Run on VisDA-C using GPU 0
bash run_visda_c.sh 0
```

### 2. Running All Datasets
To evaluate the method across all four datasets sequentially, simply execute the `run_all.sh` script:

```bash
# Run all datasets sequentially on GPU 0
bash run_all.sh 0
```

## Outputs and Logs

During execution, you will see real-time progress on the terminal. Detailed outputs are routed to specific directories:
- **Diagnostic Logs & Outputs**: Saved in the `outer_log/` directory, separated by dataset and domain pair (e.g., `outer_log/office-home/a2c.log`).
- **Performance Summaries**: Accuracy, zero-shot performance, and per-class metrics are automatically appended to a CSV file in the `result/` directory (e.g., `result/office-home_results.csv`).
- **Model Checkpoints**: Saved in `result/checkpoints/{dataset}/` after the adaptation finishes.

## Model Terminology

To align with the methodology described in our paper, the codebase has been structured with the following terminology:
- **Target Model (`T`)**: Refers to the main adaptation model (e.g., ResNet-50 backbone) that learns from the target domain.
- **Peer Model (`P`)**: Refers to the auxiliary vision-language model (e.g., CLIP with Prompt Tuning) that provides cross-architecture consistency guidance.

## Dependencies

Please ensure you have a standard PyTorch environment setup. The primary dependencies are:
- `torch`, `torchvision`
- `numpy`, `pandas`
- OpenAI's `clip`

*(Note: Deprecation warnings from PyTorch and external packages have been explicitly suppressed in `train.py` to keep the logs clean and readable for reviewers).*
