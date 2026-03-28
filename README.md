# AutoRev: Analyzing LLM-Generated vs. Human-Written Marathi Text

This repository contains the codebase and data artifacts for investigating the linguistic differences between human-written and LLM-generated Marathi text. We analyze phenomena such as lexical variance and morphological collapse by leveraging handcrafted linguistic features, Principal Component Analysis (PCA), and Support Vector Machines (SVM).

## Project Overview

The core objective of this project is to distinguish between human and machine-generated (Llama-3.2) text in a morphologically rich, low-resource language (Marathi). We extract 18 handcrafted linguistic features covering:
- **Lexical Diversity**: Type-Token Ratio, Hapax Ratio, Average Word Length.
- **Predictability & Vocabulary**: Top-100/1000 vocab overlap (measured using BPEmb reference baselines).
- **POS Density**: Frequency ratios of Nouns, Verbs, Adjectives, Adverbs, and Proper Nouns using Stanza.
- **Structural Constraints**: Dependency tree depth, POS bigram entropy, dependency pattern diversity, and the adherence to Subject-Object-Verb (SOV) sentence order.

We compare the predictive power of these explicitly engineered features with a dense embedding baseline (`BERT`), utilizing dimensionality reduction (`PCA`) to interpret which features act as the strongest discriminators for Research Question 2 (RQ2).

## Repository Structure

### Core Pipeline Scripts

- `llm_generate_data_v2.py`: Automatically generates synthetic Marathi articles based on real headline/category prompts from the test set. Uses `vLLM` to run `meta-llama/Llama-3.2-3B-Instruct` efficiently in 16-bit precision using batched inference.
- `pca_svm.py`: The main feature extraction and experimental pipeline. 
  - Uses multiprocessing to parse Marathi text with `stanza` and tokenizes with `BPEmb`.
  - Calculates the 18 specific syntactic and morphological features.
  - Extracts reference contextual embeddings from `BERT`.
  - Fits PCA models to visualize data separability and trains SVM classifiers.
- `analyze_results.py`: Post-hoc analysis script focused on evaluating **RQ2**. Produces comprehensive distribution visualizations and formatted text reports explaining feature correlations, PCA coordinate loadings, and category comparison density plots.

### Data & Results

- **Data Inputs:**
  - `SHC_Test.csv`: Base truth test split containing headlines and categories used for prompting.
  - `articles_human.csv`: Control dataset representing organic Marathi phrasing.
  - `generated_articles_v2.csv`: The primary output from the Llama generation script.
  - `handcrafted_features.csv`: Matrix of the 18 extracted dimensions per document.

- **Generated Artifacts & Visualizations:**
  - `rq2_analysis_report.txt`: A detailed summary evaluating feature contributions and PCA variance explanations.
  - `feature_correlation_matrix.png`, `feature_distributions.png`: Visual analyses of extracted statistical metrics.
  - `pca_scatter_enhanced.png`, `rq2_group_contributions.png`, `rq2_loadings_decomposition.png`: Graphic representation of how features load into the primary principal components.
  - `method_comparison_detailed.png`: Side-by-side performance comparison against base embeddings and full-feature SVM bounds.

## Setup & Execution

### Prerequisites

Ensure you have a GPU environment configured (e.g., CUDA compatibility is required for vLLM). Install the following core dependencies:

```bash
pip install pandas torch huggingface_hub vllm stanza bpemb scikit-learn matplotlib seaborn
```

### Steps to Run

1. **Text Generation**: You will need a Hugging Face token with access to Llama 3.2. Specify your token in `llm_generate_data_v2.py`, then generate the dataset:
   ```bash
   python llm_generate_data_v2.py
   ```
2. **Feature Extraction & Classification**: Compute dependencies, tree structures, fit PCA, and evaluate the SVM vs. BERT baseline:
   ```bash
   python pca_svm.py
   ```
   *(Note: Stanza will run a one-time download of the Marathi `mr` models on initial launch.)*

3. **Post-Hoc Analysis**: To generate visualizations interpreting the impact of different feature groups and saving out the `rq2_analysis_report.txt`:
   ```bash
   python analyze_results.py
   ```
