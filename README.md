# Safeguarding Mutual Correction in Source-Free Domain Adaptation via Cut Statistics

## 🎉**Accepted at NeurIPS 2026.**🎉

This repository contains the official PyTorch implementation of our NeurIPS 2026 paper, *Safeguarding Mutual Correction in Source-Free Domain Adaptation via Cut Statistics*.

**Seongjun Lee · Changhee Lee\***<br>
Korea University<br>
\* Corresponding author

## 📁 Project Structure

```text
SafeCut/
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

## Dataset and Pre-trained Model Setup

The datasets and pre-trained source models are not distributed with this repository. Please download them separately and configure their local paths before running SafeCut.

### Datasets

Download Office-Home, Office-31, DomainNet-126, and VisDA-C from their respective official sources. The image-list files under `data/` use the following format:

```text
/absolute/path/to/an/image.jpg class_index
```

Before running an experiment, update the first column of the corresponding `*_list.txt` files so that every entry points to the dataset location on your machine:

- `data/office-home/*_list.txt`
- `data/office/*_list.txt`
- `data/domainnet126/*_list.txt`
- `data/VISDA-C/*_list.txt`

### Pre-trained Source Models

For the pre-trained source models used in this repository, we referred to the [source-free-domain-adaptation](https://github.com/tntek/source-free-domain-adaptation) repository. After obtaining the weights, place them under `pretrained_model/` using the following source-domain directories:

| Dataset | Required source-domain directories |
| --- | --- |
| Office-Home | `office-home/{A,C,P,R}/` |
| Office-31 | `office/{A,D,W}/` |
| DomainNet-126 | `domainnet126/{C,P,R,S}/` |
| VisDA-C | `VISDA-C/T/` |

Each source-domain directory must contain all three component checkpoints:

```text
source_F.pt
source_B.pt
source_C.pt
```

Both `data/` and `pretrained_model/` are resolved relative to the current working directory. Run all commands below from the repository root.

## Execution Guide

We provide simple shell scripts to easily run the SafeCut adaptation process across different benchmark datasets. 

### 1. Running a Single Dataset
You can run the adaptation on a specific dataset using its dedicated execution script. The scripts take the `GPU_ID` as an optional argument (defaults to `0`).

```bash
# Run on Office-Home using GPU 0
bash run_office_home.sh 0

# Run on Office-31 using GPU 0
bash run_office31.sh 0

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

## Contact

For questions, please contact [Seongjun Lee](mailto:pyoung7307@korea.ac.kr) or [Changhee Lee](mailto:changheelee@korea.ac.kr).

## Installation

We recommend creating a dedicated Conda environment named `SafeCut`:

```bash
conda create -n SafeCut python=3.12 -y
conda activate SafeCut
```

Install a PyTorch build compatible with your CUDA environment by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Our experiments were run in an internal Conda environment named `TTA` with the following versions:

- Python 3.12.12
- PyTorch 2.9.0 with CUDA 13.0
- torchvision 0.24.0

During release preparation, dependency imports and CLI initialization were verified with this configuration. The Conda environment name itself does not affect execution.

## License

The original SafeCut contributions are released under the [MIT License](LICENSE). Third-party components retain their respective licenses; see [Third-Party Notices](THIRD_PARTY_NOTICES.md) for details.
