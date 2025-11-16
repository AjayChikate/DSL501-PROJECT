# DSL501-PROJECT

## Dataset

We use the [lmsys-chat-1m dataset](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).  
Due to its large size (1 million entries), we showcase only the first 500 entries.  
You can retrieve the full dataset as described below.

---

## Accessing Gated Datasets (Hugging Face CLI Login)

The `lmsys-chat-1m` dataset requires authentication for download or programmatic access.  
Follow these steps to log in using the Hugging Face Command Line Interface (CLI):

### A. Terminal Login

1. **Open your terminal** . 
2. **Run the login command:**
    ```bash
    huggingface-cli login
    ```
3. Enter your **Hugging Face access token** when prompted.

### B. Generating an Access Token

If you do not have a token:

1. Go to the [Hugging Face website](https://huggingface.co/) and log into your account.
2. Navigate to **Settings** (click your profile picture).
3. Click **Access Tokens** in the sidebar.
4. Click **+ New token**.
5. Provide a name and select the required role (e.g., 'Read' or 'Write/Read').
6. **Copy the generated token** and paste it into your terminal when prompted by `huggingface-cli login`.

---

After logging in, run `DatasetRetrieval.ipynb` to obtain the data in separate train and test splits.

---

## Setup

To get started with the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/farhan11166/DSL501-PROJECT.git
    cd DSL501-PROJECT
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Prepare the Dataset

After logging in to Hugging Face CLI (as described above) and installing dependencies, run the `DatasetRetrieval.ipynb` notebook to download and prepare the dataset. This will create the necessary train and test splits.

Alternatively, you can use the `prepare_dataset.py` script:

```bash
python scripts/prepare_dataset.py
```

### 2. Train the Model

To train the PMPO model, use the `train.py` script:

```bash
python scripts/train.py --config_path configs/default.yaml
```

You can modify `configs/default.yaml` to adjust training parameters.

### 3. Experiment with Jupyter Notebooks

Explore various experimental setups and PMPO configurations using the provided Jupyter notebooks:

*   `PMPO_ALPHA=...GPT_NEO.ipynb`
*   `PMPO_On_GPT2.ipynb`
*   `environments.ipynb`
*   `Evaluation.ipynb`
*   `PreprocessingForPMPO.ipynb`
*   `PMPO_usingKL_Divergence.ipynb`

---


> **Note:**  
> For detailed information about the project, please refer to [SoP_ML.pdf](https://github.com/farhan11166/DSL501-PROJECT/blob/main/SoP_ML.pdf).

## Codebase Overview

This project implements Preference-based Maximum a Posteriori Optimization (PMPO) for fine-tuning causal language models, primarily for tasks involving human preferences.

### Key Components:

*   **Data Handling (`pmpo/data.py`, `Dataset/`, `scripts/prepare_dataset.py`):**
    *   The `PreferenceDataset` class (in `pmpo/data.py`) is responsible for loading and tokenizing preference data from CSV files. It expects a 'label' column to differentiate between positive and negative examples.
    *   The `Dataset/` directory and `scripts/prepare_dataset.py` are dedicated to the preparation and organization of these datasets.

*   **PMPO Core (`pmpo/trainer.py`):**
    *   The `PMPOTrainer` class (in `pmpo/trainer.py`) encapsulates the core PMPO algorithm.
    *   It leverages pre-trained causal language models (e.g., GPT-2) from the Hugging Face `transformers` library.
    *   The training objective is designed to balance three key terms: a positive term (to increase the likelihood of preferred responses), a negative term (to decrease the likelihood of dispreferred responses), and a KL divergence term (to prevent the model from drifting too far from a reference model).
    *   For enhanced efficiency, the trainer incorporates features such as gradient checkpointing and mixed-precision training.

*   **Training Workflow (`scripts/train.py`, Jupyter Notebooks):**
    *   The `scripts/train.py` script orchestrates the end-to-end training process, including data loading, initialization of the PMPO trainer, and execution of the optimization loop.
    *   Various Jupyter notebooks (e.g., `PMPO_ALPHA=...GPT_NEO.ipynb`, `PMPO_On_GPT2.ipynb`) are provided for experimental setups, allowing for exploration of different PMPO configurations and model architectures.

*   **Configuration (`configs/default.yaml`):**
    *   The `configs/default.yaml` file is designated for storing configuration parameters pertinent to the training process.

*   **Dependencies (`requirements.txt`):**
    *   The project relies on a set of standard deep learning libraries, including `torch`, `transformers`, `datasets`, `tokenizers`, `tqdm`, and `numpy`.
