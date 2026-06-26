# Software Engineering (SE) Domain Specific Embeddings: Comparative Study. 
Title: ModernBERT for Software Engineering: Domain Adaptation, Contextual Embedding, and Empirical Trade-off Analysis

## Overview

This work implements a comprehensive framework for the collection, preprocessing, training, and evaluation of word embedding models specifically tailored for the Software Engineering (SE) domain. We compare the traditional static embedding models viz. (Word2Vec, gloVe and fastText) with transformer-based approaches (ModernBERT) to understand their performance, computational trade-offs, and applicability in various SE tasks.

## Authors
Anonymous

## Key Features

- **Automated Data Collection:** Gathers authentic SE documents from Wikipedia, GitHub, Stack Overflow, and ArXiv.
- **Rigorous Preprocessing Pipeline:** Ensures high-quality data through filtering, cleaning, and domain-specific tokenization.
- **Static Word Embedding Model Implementation for SE domain:** Baseline models training and evaluation for SE contexts.
- **ModernBERT Implementation for SE domain:** Transformer-based model training and comparative analysis (including expected performance and challenges).
- **Comprehensive Evaluation Framework:** Intrinsic (word similarity, analogies, vocabulary coverage) and extrinsic (classification, clustering) metrics.
- **Computational Analysis:** Benchmarking of training time, memory usage, and inference speed for both models.
- **Reproducible Methodology:** Detailed configurations and setup for replication of experiments.

## Research Questions Addressed

1.  How effectively do transformer-based embeddings (e.g., ModernBERT) capture context-dependent meanings in SE terminology compared to static embeddings such as Word2Vec, fastText, and GloVe??
2.  What are the computational and performance trade-offs between transformer-based embeddings (ModernBERT) and static embeddings, when applied to SE tasks?
3.  What implementation challenges arise when adapting transformer models to SE domains, and how can they be mitigated through structured methodologies?
4.  How do different word embedding approaches perform across SE tasks such as requirements analysis, documentation processing, and technical terminology representation?

## Quick Start

Follow these steps to set up and run the repo locally.

### Prerequisites

- Python 3.9+ (recommended)
- Git
- GitHub Desktop (for easy drag-and-drop workflow)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/SE-Word-Embeddings-Project.git
    cd SE-Word-Embeddings-Project
    ```
    *(If using GitHub Desktop, clone via the application and then navigate to the folder. )*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

This project supports two main configurations: a quick test run and a full-scale production run.

-   **Quick Test Run (approx. 5-10 minutes):**
    ```bash
    python main.py --config config/test.yaml
    ```
    *This will collect a small sample of real data and run a minimal training/evaluation to verify setup.*

-   **Full-Scale Production Run (approx. 4-6 hours):**
    ```bash
    python main.py --config config/fullscale.yaml
    ```
    *This will collect a large volume of real data, train models comprehensively, and perform full evaluations.*

## Project Structure

```text
SE_WE/
├── config/                         # Configuration files for execution
│   ├── test.yaml                   # Fast pipeline verification run (test file)
│   └── fullscale.yaml              # Complete run file
├── results/                        # All pipeline-generated outputs
│   ├── data/
│   │   ├── raw/                    # Collected raw domain data
│   │   └── processed/              # Tokenized, cleaned, and split data
│   ├── models/                     # Saved local model weights & layers
│   │   ├── modernbert/             # ModernBERT tokenizer and model files
│   │   └── word2vec/               # Saved Word2Vec embedding weights
│   ├── evaluations/                # Raw metric outputs & tracking logs
│   │   ├── baseline_comparison.json # Baseline evaluations (GloVe, FastText)
│   │   └── comprehensive_evaluation.json # Evaluation metrics
│   ├── reports/                    # Texts
│   │   └── evaluation_summary.txt  # Terminal text summary
│   └── logs/                       # System execution trace files
│       └── se_embeddings.log       # Debugging and run tracking data
├── src/                            # Source code repository files
│   ├── models/                     # Model-specific tracking scripts
│   │   ├── word2vec_model.py       # Word2Vec lifecycle execution
│   │   └── modernbert_model.py     # ModernBERT Transformer setup
│   ├── evaluate_fasttext_glove.py  # Static baseline evaluation loop
│   └── main.py                     # Primary project orchestration runner
|   └── train_save_fasttext         # Most Similar words for fastText and GloVe
├── venv/                           # Isolated local Python environment (Ignored)
├── .gitignore                      # Git exclusion filters (weights, venv, etc.)
└── README.md                       # Documentation


*Note: ModernBERT results are based on successful implementation and evaluation within the project, reflecting the comparative analysis.* 

## License

This work is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
