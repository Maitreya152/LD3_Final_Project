"""
AutoRev: Analyzing Lexical Variance and Morphological Collapse
in LLM-Generated vs. Human-Written Marathi Text via PCA.

Full pipeline: Feature Extraction → PCA → SVM → BERT Baseline → Combined Study
All results, models, plots, and logs are saved to a timestamped results directory.
"""

import os
import sys
import argparse
import json
import logging
import datetime
import string
from collections import Counter
from itertools import islice

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import joblib

# NLP Libraries
from indicnlp.tokenize import indic_tokenize, sentence_tokenize
import stanza
from bpemb import BPEmb

# Scikit-Learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Deep Learning (for BERT baseline)
import torch
from transformers import AutoTokenizer, AutoModel

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
HUMAN_CSV = "articles_human.csv"
LLM_CSV = "generated_articles_v2.csv"
BERT_MODEL_NAME = "l3cube-pune/marathi-bert"
PCA_COMPONENTS = 2
SVM_KERNEL = "linear"
SVM_C = 1.0
TEST_SIZE = 0.2
RANDOM_STATE = 42
BERT_BATCH_SIZE = 32
BERT_MAX_LENGTH = 512
TOP_K_VOCAB = (100, 1000)


# ═══════════════════════════════════════════════════════════════
# RESULTS DIRECTORY SETUP
# ═══════════════════════════════════════════════════════════════
def setup_results_dir():
    """Creates a timestamped results directory with subdirectories."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("results", f"run_{timestamp}")
    subdirs = ["features", "pca", "models", "metrics", "plots", "logs"]
    for sub in subdirs:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
    return base_dir


def setup_logging(results_dir):
    """Configures dual logging to file and console."""
    log_path = os.path.join(results_dir, "logs", "pipeline.log")
    logger = logging.getLogger("autorev")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ═══════════════════════════════════════════════════════════════
# MULTIPROCESSING WORKER SETUP
# ═══════════════════════════════════════════════════════════════
global_nlp_stanza = None
global_bpemb_mr = None
global_top_100 = None
global_top_1000 = None


def init_worker(top_100, top_1000):
    """Initializes the heavy NLP models once per CPU core."""
    global global_nlp_stanza, global_bpemb_mr, global_top_100, global_top_1000
    global_nlp_stanza = stanza.Pipeline(
        'mr', processors='tokenize,pos,lemma,depparse', use_gpu=False, verbose=False
    )
    global_bpemb_mr = BPEmb(lang="mr", vs=10000)
    global_top_100 = top_100
    global_top_1000 = top_1000


def compute_reference_vocab(human_df, top_k=TOP_K_VOCAB):
    """Computes the most common words from the human corpus as a baseline."""
    all_words = []
    for text in tqdm(human_df['Text'].dropna(), desc="Building reference vocabulary"):
        tokens = indic_tokenize.trivial_tokenize(str(text))
        all_words.extend([t for t in tokens if t not in string.punctuation])

    freq_dist = Counter(all_words)
    top_100 = set([word for word, _ in freq_dist.most_common(top_k[0])])
    top_1000 = set([word for word, _ in freq_dist.most_common(top_k[1])])
    return top_100, top_1000


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (18 Handcrafted Features)
# ═══════════════════════════════════════════════════════════════
def _compute_tree_depth(sentence):
    """Computes the maximum depth of the dependency tree for a stanza sentence."""
    children = {}
    root_id = None
    for word in sentence.words:
        if word.head == 0:
            root_id = word.id
        children.setdefault(word.head, []).append(word.id)

    if root_id is None:
        return 0

    depth = 0
    queue = [(root_id, 1)]
    while queue:
        node_id, d = queue.pop(0)
        depth = max(depth, d)
        for child_id in children.get(node_id, []):
            queue.append((child_id, d + 1))
    return depth


def _compute_sov_ratio(doc):
    """
    Estimates the Subject-Object-Verb order ratio for Marathi (SOV language).
    For each sentence: checks if nsubj appears before obj which appears before the verb root.
    """
    sov_count = 0
    total_valid = 0

    for sentence in doc.sentences:
        subj_pos = None
        obj_pos = None
        verb_pos = None

        for word in sentence.words:
            if word.deprel in ('nsubj', 'nsubj:pass') and subj_pos is None:
                subj_pos = word.id
            elif word.deprel in ('obj', 'iobj') and obj_pos is None:
                obj_pos = word.id
            if word.upos == 'VERB' and word.deprel == 'root':
                verb_pos = word.id

        if subj_pos is not None and obj_pos is not None and verb_pos is not None:
            total_valid += 1
            if subj_pos < obj_pos < verb_pos:
                sov_count += 1

    return sov_count / total_valid if total_valid > 0 else 0.0


def process_single_row(row_dict):
    """Worker function to extract all 18 handcrafted features for a single text."""
    text = str(row_dict['Text']).strip()
    label = row_dict['Source']
    category = str(row_dict.get('Label', 'Unknown')).strip()

    if not text:
        return None, None, None

    features = {}

    sentences = sentence_tokenize.sentence_split(text, lang='mr')
    tokens = indic_tokenize.trivial_tokenize(text)
    words = [t for t in tokens if t not in string.punctuation]

    if len(words) == 0 or len(sentences) == 0:
        return None, None, None

    # ── Lexical Features ─────────────────────────────────────
    unique_words = set(words)
    word_freq = Counter(words)
    features['type_token_ratio'] = len(unique_words) / len(words)
    features['hapax_ratio'] = sum(1 for _, count in word_freq.items() if count == 1) / len(words)
    features['avg_word_length'] = np.mean([len(w) for w in words])

    # ── Predictability ───────────────────────────────────────
    features['top_100_overlap_pct'] = sum(1 for w in words if w in global_top_100) / len(words)
    features['top_1000_overlap_pct'] = sum(1 for w in words if w in global_top_1000) / len(words)

    # ── Basic Syntactic Features ─────────────────────────────
    sent_lengths = [len(indic_tokenize.trivial_tokenize(s)) for s in sentences]
    features['avg_sentence_length'] = np.mean(sent_lengths)
    features['sentence_length_variance'] = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
    features['punctuation_density'] = sum(1 for char in text if char in string.punctuation) / len(tokens)

    # ── Morphological Complexity ─────────────────────────────
    bpe_tokens = global_bpemb_mr.encode(text)
    features['morphological_complexity'] = len(bpe_tokens) / len(words)

    # ── Stanza Deep Syntax ───────────────────────────────────
    doc = global_nlp_stanza(text)

    dependency_lengths = []
    pos_tags = []
    prefixes = []
    postfixes = []
    dep_triples = []       # (head_upos, deprel, child_upos)
    tree_depths = []
    pos_tag_counter = Counter()

    for sentence_obj in doc.sentences:
        # Tree depth
        tree_depths.append(_compute_tree_depth(sentence_obj))

        for word in sentence_obj.words:
            pos_tags.append(word.upos)
            pos_tag_counter[word.upos] += 1

            # Prefix / Postfix
            if len(word.text) > 4:
                prefixes.append(word.text[:2])
                postfixes.append(word.text[-2:])

            # Dependency length
            if word.head > 0:
                head_word = sentence_obj.words[word.head - 1]
                dependency_lengths.append(abs(word.id - head_word.id))
                dep_triples.append((head_word.upos, word.deprel, word.upos))

    total_pos = len(pos_tags) if len(pos_tags) > 0 else 1

    # Existing syntactic features
    features['avg_dependency_length'] = np.mean(dependency_lengths) if dependency_lengths else 0
    features['unique_pos_ratio'] = len(set(pos_tags)) / total_pos
    features['prefix_variance'] = len(set(prefixes)) / len(prefixes) if prefixes else 0
    features['postfix_variance'] = len(set(postfixes)) / len(postfixes) if postfixes else 0

    # ── NEW: POS Density Features ────────────────────────────
    features['noun_ratio'] = pos_tag_counter.get('NOUN', 0) / total_pos
    features['verb_ratio'] = pos_tag_counter.get('VERB', 0) / total_pos
    features['adj_ratio'] = pos_tag_counter.get('ADJ', 0) / total_pos
    features['adv_ratio'] = pos_tag_counter.get('ADV', 0) / total_pos
    features['propn_ratio'] = pos_tag_counter.get('PROPN', 0) / total_pos

    # ── NEW: POS Bigram Entropy ──────────────────────────────
    if len(pos_tags) > 1:
        bigrams = [(pos_tags[i], pos_tags[i + 1]) for i in range(len(pos_tags) - 1)]
        bigram_counts = Counter(bigrams)
        total_bigrams = sum(bigram_counts.values())
        probs = np.array([c / total_bigrams for c in bigram_counts.values()])
        features['pos_bigram_entropy'] = -np.sum(probs * np.log2(probs + 1e-12))
    else:
        features['pos_bigram_entropy'] = 0.0

    # ── NEW: Dependency Pattern Diversity ────────────────────
    if dep_triples:
        features['dep_pattern_diversity'] = len(set(dep_triples)) / len(dep_triples)
    else:
        features['dep_pattern_diversity'] = 0.0

    # ── NEW: Average Tree Depth ──────────────────────────────
    features['tree_depth_avg'] = np.mean(tree_depths) if tree_depths else 0.0

    # ── NEW: SOV Word Order Ratio ────────────────────────────
    features['sov_ratio'] = _compute_sov_ratio(doc)

    return features, label, category


# ═══════════════════════════════════════════════════════════════
# BERT EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════
def extract_bert_embeddings(texts, model_name, batch_size=BERT_BATCH_SIZE, max_length=BERT_MAX_LENGTH):
    """Extracts [CLS] token embeddings from a BERT model for a list of texts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"BERT embeddings ({device})"):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # [CLS] token is the first token
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


# ═══════════════════════════════════════════════════════════════
# METRICS & EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════
def calculate_dual_metrics(y_true, y_predict):
    """Calculates accuracy, precision, recall, F1 for both classes."""
    acc = accuracy_score(y_true, y_predict)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_predict, labels=[0, 1], zero_division=0
    )
    return {
        'Accuracy': float(acc),
        'Human_Precision': float(prec[0]), 'Human_Recall': float(rec[0]),
        'Human_F1': float(f1[0]), 'Human_Support': int(support[0]),
        'LLM_Precision': float(prec[1]), 'LLM_Recall': float(rec[1]),
        'LLM_F1': float(f1[1]), 'LLM_Support': int(support[1]),
    }


def run_pca_svm_experiment(X, y, categories, label, results_dir, logger, save_models=False):
    """
    Runs the full PCA → SVM pipeline on a feature matrix.
    Returns (global_metrics, category_metrics_df, X_pca, pca, svm, scaler, X_train, X_test, y_train, y_test, cat_test).
    """
    logger.info(f"[{label}] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"[{label}] Applying PCA (n_components={PCA_COMPONENTS})...")
    pca = PCA(n_components=PCA_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)

    ev = pca.explained_variance_ratio_
    logger.info(f"[{label}] Variance: PC1={ev[0]*100:.2f}%, PC2={ev[1]*100:.2f}%")

    logger.info(f"[{label}] Training Linear SVM (C={SVM_C})...")
    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X_pca, y, categories, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    svm_clf = SVC(kernel=SVM_KERNEL, C=SVM_C)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    global_metrics = calculate_dual_metrics(y_test, y_pred)
    logger.info(f"[{label}] Global Accuracy: {global_metrics['Accuracy']:.4f}")

    # Category-wise
    cat_results = []
    for cat in np.unique(cat_test):
        mask = (cat_test == cat)
        if sum(mask) == 0:
            continue
        metrics = calculate_dual_metrics(y_test[mask], y_pred[mask])
        metrics['Category'] = cat
        metrics['Total_Support'] = int(sum(mask))
        cat_results.append(metrics)

    cat_df = pd.DataFrame(cat_results)
    if not cat_df.empty:
        cat_df = cat_df.sort_values(by='LLM_F1', ascending=True)

    # Save models if this is the primary (handcrafted) run
    if save_models:
        joblib.dump(scaler, os.path.join(results_dir, "models", "scaler.joblib"))
        joblib.dump(pca, os.path.join(results_dir, "models", "pca_model.joblib"))
        joblib.dump(svm_clf, os.path.join(results_dir, "models", "svm_model.joblib"))
        logger.info(f"[{label}] Models saved to {results_dir}/models/")

    return global_metrics, cat_df, X_pca, pca, svm_clf, scaler, X_train, X_test, y_train, y_test, cat_test


def run_svm_no_pca_experiment(X, y, categories, label, results_dir, logger):
    """
    Runs SVM directly on scaled features WITHOUT PCA dimensionality reduction.
    Returns (global_metrics, category_metrics_df).
    """
    logger.info(f"[{label}] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"[{label}] Training Linear SVM on full {X_scaled.shape[1]} features (C={SVM_C})...")
    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X_scaled, y, categories, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    svm_clf = SVC(kernel=SVM_KERNEL, C=SVM_C)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    global_metrics = calculate_dual_metrics(y_test, y_pred)
    logger.info(f"[{label}] Global Accuracy: {global_metrics['Accuracy']:.4f}")

    cat_results = []
    for cat in np.unique(cat_test):
        mask = (cat_test == cat)
        if sum(mask) == 0:
            continue
        metrics = calculate_dual_metrics(y_test[mask], y_pred[mask])
        metrics['Category'] = cat
        metrics['Total_Support'] = int(sum(mask))
        cat_results.append(metrics)

    cat_df = pd.DataFrame(cat_results)
    if not cat_df.empty:
        cat_df = cat_df.sort_values(by='LLM_F1', ascending=True)

    # Save metrics
    with open(os.path.join(results_dir, 'metrics', f'{label.lower()}_global.json'), 'w') as f:
        json.dump(global_metrics, f, indent=2)
    cat_df.to_csv(os.path.join(results_dir, 'metrics', f'{label.lower()}_category.csv'), index=False)

    # Save model
    joblib.dump(svm_clf, os.path.join(results_dir, 'models', f'{label.lower()}_svm.joblib'))
    joblib.dump(scaler, os.path.join(results_dir, 'models', f'{label.lower()}_scaler.joblib'))

    return global_metrics, cat_df


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════
def generate_all_plots(results_dir, X_pca, y, categories, pca, feature_names,
                       cat_df, X_test, y_test, cat_test, comparison_df, logger):
    """Generates and saves all visualization plots."""
    plot_dir = os.path.join(results_dir, "plots")
    sns.set_theme(style="whitegrid")
    ev = pca.explained_variance_ratio_

    # ── 1. PCA Global Scatter ────────────────────────────────
    logger.info("Generating PCA global scatter plot...")
    plt.figure(figsize=(10, 8))
    hue_labels = ['LLM' if l == 1 else 'Human' for l in y]
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hue_labels,
                    palette=['#1f77b4', '#ff7f0e'], alpha=0.6, s=50)
    plt.title(f'Global PCA: Human vs LLM\n(PC1: {ev[0]*100:.1f}%, PC2: {ev[1]*100:.1f}%)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(plot_dir, 'pca_distribution_global.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ── 2. Category-wise F1 Bar Chart ────────────────────────
    if not cat_df.empty:
        logger.info("Generating category-wise F1 bar chart...")
        plt.figure(figsize=(14, 8))
        x_indices = np.arange(len(cat_df))
        width = 0.35
        plt.bar(x_indices - width / 2, cat_df['Human_F1'], width,
                label='Human Text F1', color='#1f77b4')
        plt.bar(x_indices + width / 2, cat_df['LLM_F1'], width,
                label='LLM Text F1', color='#ff7f0e')
        plt.xlabel('News Category')
        plt.ylabel('F1 Score')
        plt.title('Category-wise F1 Scores (Human vs. LLM Detection)')
        plt.xticks(x_indices, cat_df['Category'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'category_f1_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ── 3. Faceted PCA by Category ───────────────────────────
    logger.info("Generating faceted PCA by category...")
    test_vis_df = pd.DataFrame({
        'PC1': X_test[:, 0], 'PC2': X_test[:, 1],
        'Source': ['LLM' if l == 1 else 'Human' for l in y_test],
        'Category': cat_test
    })
    g = sns.FacetGrid(test_vis_df, col="Category", col_wrap=3, height=4,
                      hue="Source", palette=['#1f77b4', '#ff7f0e'])
    g.map_dataframe(sns.scatterplot, x="PC1", y="PC2", alpha=0.7)
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('PCA 2D Projection by News Category', fontsize=16)
    plt.savefig(os.path.join(plot_dir, 'pca_by_category.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ── 4. PCA Loadings Heatmap ──────────────────────────────
    logger.info("Generating PCA loadings heatmap...")
    loadings = pca.components_.T  # shape: (n_features, n_components)
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2'])

    plt.figure(figsize=(8, max(6, len(feature_names) * 0.4)))
    sns.heatmap(loadings_df, annot=True, fmt=".3f", cmap='RdBu_r', center=0,
                linewidths=0.5, cbar_kws={"label": "Loading"})
    plt.title('PCA Component Loadings (Handcrafted Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'pca_loadings_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ── 5. Method Comparison Bar Chart ───────────────────────
    if comparison_df is not None and not comparison_df.empty:
        logger.info("Generating method comparison bar chart...")
        plt.figure(figsize=(14, 6))
        x_idx = np.arange(len(comparison_df))
        width = 0.25
        plt.bar(x_idx - width, comparison_df['Accuracy'], width, label='Accuracy', color='#2ca02c')
        plt.bar(x_idx, comparison_df['Human_F1'], width, label='Human F1', color='#1f77b4')
        plt.bar(x_idx + width, comparison_df['LLM_F1'], width, label='LLM F1', color='#ff7f0e')
        plt.xlabel('Method')
        plt.ylabel('Score')
        plt.title('Method Comparison')
        plt.xticks(x_idx, comparison_df['Method'], rotation=30, ha='right', fontsize=9)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'method_comparison_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ═══════════════════════════════════════════════════════════════
# SUMMARY GENERATION
# ═══════════════════════════════════════════════════════════════
def generate_summary(results_dir, dataset_stats, feature_names, pca, ev,
                     hc_metrics, full_feat_metrics, bert_metrics, bert_full_metrics,
                     combined_metrics, hc_cat_df, bert_cat_df, combined_cat_df, logger):
    """Generates and saves a comprehensive text summary."""
    lines = []

    def add(text=""):
        lines.append(text)
        logger.info(text)

    add("=" * 80)
    add("  AutoRev — FULL PIPELINE SUMMARY")
    add(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("=" * 80)

    add("\n📊 DATASET STATISTICS")
    add("-" * 40)
    for k, v in dataset_stats.items():
        add(f"  {k}: {v}")

    add(f"\n🔬 FEATURES EXTRACTED ({len(feature_names)} total)")
    add("-" * 40)
    for i, name in enumerate(feature_names, 1):
        add(f"  {i:2d}. {name}")

    add(f"\n📐 PCA EXPLAINED VARIANCE")
    add("-" * 40)
    add(f"  PC1: {ev[0]*100:.2f}%")
    add(f"  PC2: {ev[1]*100:.2f}%")
    add(f"  Total: {sum(ev)*100:.2f}%")

    add(f"\n🏷️  TOP PCA LOADINGS (Research Question 2)")
    add("-" * 40)
    loadings_df = pd.DataFrame(
        pca.components_.T, index=feature_names, columns=['PC1', 'PC2']
    )
    for pc in ['PC1', 'PC2']:
        sorted_loadings = loadings_df[pc].abs().sort_values(ascending=False)
        add(f"  {pc} — Top 5 features by |loading|:")
        for feat_name, val in sorted_loadings.head(5).items():
            sign = "+" if loadings_df.loc[feat_name, pc] > 0 else "-"
            add(f"    {sign}{val:.4f}  {feat_name}")

    add("\n" + "=" * 80)
    add("📈 CLASSIFICATION RESULTS")
    add("=" * 80)

    methods = [
        ("HC Features \u2192 PCA \u2192 SVM", hc_metrics),
        ("HC Features \u2192 SVM", full_feat_metrics),
        ("BERT Embeddings \u2192 PCA \u2192 SVM", bert_metrics),
        ("BERT Embeddings \u2192 SVM", bert_full_metrics),
        ("HC + BERT Combined \u2192 PCA \u2192 SVM", combined_metrics),
    ]

    for method_name, metrics in methods:
        if metrics is None:
            add(f"\n  [{method_name}] — SKIPPED")
            continue
        add(f"\n  [{method_name}]")
        add(f"    Overall Accuracy:  {metrics['Accuracy']:.4f}")
        add(f"    HUMAN → P: {metrics['Human_Precision']:.4f}  R: {metrics['Human_Recall']:.4f}  F1: {metrics['Human_F1']:.4f}")
        add(f"    LLM   → P: {metrics['LLM_Precision']:.4f}  R: {metrics['LLM_Recall']:.4f}  F1: {metrics['LLM_F1']:.4f}")

    add("\n" + "=" * 80)
    add("🔀 COMPARATIVE SUMMARY")
    add("=" * 80)
    comp_header = f"  {'Method':<35} {'Accuracy':>10} {'Human F1':>10} {'LLM F1':>10}"
    add(comp_header)
    add("  " + "-" * 67)
    for method_name, metrics in methods:
        if metrics is None:
            add(f"  {method_name:<35} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            add(f"  {method_name:<35} {metrics['Accuracy']:>10.4f} {metrics['Human_F1']:>10.4f} {metrics['LLM_F1']:>10.4f}")

    add("\n" + "=" * 80)
    add("📂 CATEGORY-WISE ANALYSIS (Handcrafted)")
    add("=" * 80)
    if hc_cat_df is not None and not hc_cat_df.empty:
        display_cols = ['Category', 'Accuracy', 'Human_F1', 'LLM_F1',
                        'Human_Precision', 'LLM_Precision',
                        'Human_Recall', 'LLM_Recall', 'Total_Support']
        available_cols = [c for c in display_cols if c in hc_cat_df.columns]
        add(hc_cat_df[available_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    add("\n" + "=" * 80)
    add("📁 RESULTS SAVED TO")
    add("=" * 80)
    add(f"  {os.path.abspath(results_dir)}")

    add("\n" + "=" * 80)

    summary_text = "\n".join(lines)
    with open(os.path.join(results_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    return summary_text


# ═══════════════════════════════════════════════════════════════
# RESUME FROM PREVIOUS RUN
# ═══════════════════════════════════════════════════════════════
def main_from_run(source_run_dir):
    """Loads cached features/embeddings from a previous run and re-runs Phases 2-6."""
    results_dir = setup_results_dir()
    logger = setup_logging(results_dir)

    logger.info(f"AutoRev Pipeline — RESUME MODE from: {os.path.abspath(source_run_dir)}")
    logger.info(f"New results will be saved to: {os.path.abspath(results_dir)}")

    # ── Load cached features ─────────────────────────────────
    features_path = os.path.join(source_run_dir, 'features', 'handcrafted_features.csv')
    if not os.path.exists(features_path):
        logger.error(f"Cannot find {features_path}. Is this a valid run directory?")
        sys.exit(1)

    logger.info("Loading cached handcrafted features...")
    df_features_full = pd.read_csv(features_path)

    # Separate metadata from features
    y = df_features_full['label'].values
    categories = df_features_full['category'].values
    df_features = df_features_full.drop(columns=['label', 'category'])
    feature_names = df_features.columns.tolist()

    logger.info(f"Loaded {len(df_features)} samples × {len(feature_names)} features")

    dataset_stats = {
        'Source run': os.path.abspath(source_run_dir),
        'Total samples': len(df_features),
        'Human samples': int(np.sum(y == 0)),
        'LLM samples': int(np.sum(y == 1)),
        'Categories': sorted(list(set(categories))),
    }

    # Save features to new run dir
    df_features_full.to_csv(os.path.join(results_dir, 'features', 'handcrafted_features.csv'), index=False)
    df_features_full.to_csv('extracted_features.csv', index=False)

    # ══════════════════════════════════════════════════════════
    # PHASE 2: PCA + SVM on Handcrafted Features
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2: PCA + SVM on handcrafted features...")
    logger.info("=" * 60)

    (hc_metrics, hc_cat_df, X_pca_hc, pca_hc, svm_hc, scaler_hc,
     X_train_hc, X_test_hc, y_train_hc, y_test_hc, cat_test_hc) = run_pca_svm_experiment(
        df_features.values, y, categories, "Handcrafted", results_dir, logger, save_models=True
    )

    ev_hc = pca_hc.explained_variance_ratio_
    loadings_df = pd.DataFrame(pca_hc.components_.T, index=feature_names, columns=['PC1', 'PC2'])
    loadings_df.to_csv(os.path.join(results_dir, 'pca', 'pca_loadings.csv'))

    ev_dict = {
        'PC1_variance_ratio': float(ev_hc[0]),
        'PC2_variance_ratio': float(ev_hc[1]),
        'total_variance_ratio': float(sum(ev_hc)),
    }
    with open(os.path.join(results_dir, 'pca', 'explained_variance.json'), 'w') as f:
        json.dump(ev_dict, f, indent=2)

    pca_coords_df = pd.DataFrame(X_pca_hc, columns=['PC1', 'PC2'])
    pca_coords_df['label'] = y
    pca_coords_df['category'] = categories
    pca_coords_df.to_csv(os.path.join(results_dir, 'pca', 'pca_transformed.csv'), index=False)

    with open(os.path.join(results_dir, 'metrics', 'handcrafted_global.json'), 'w') as f:
        json.dump(hc_metrics, f, indent=2)
    hc_cat_df.to_csv(os.path.join(results_dir, 'metrics', 'handcrafted_category.csv'), index=False)

    # ══════════════════════════════════════════════════════════
    # PHASE 2b: Full Features SVM (No PCA)
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2b: SVM on full feature set (no PCA)...")
    logger.info("=" * 60)

    full_feat_metrics, full_feat_cat_df = run_svm_no_pca_experiment(
        df_features.values, y, categories, "full_features", results_dir, logger
    )

    # ══════════════════════════════════════════════════════════
    # PHASE 3, 3b, 4: BERT PCA + BERT Full + Combined (from cache)
    # ══════════════════════════════════════════════════════════
    bert_metrics = None
    bert_cat_df = None
    bert_full_metrics = None
    bert_full_cat_df = None
    combined_metrics = None
    combined_cat_df = None

    bert_path = os.path.join(source_run_dir, 'features', 'bert_embeddings.npy')
    if os.path.exists(bert_path):
        logger.info("=" * 60)
        logger.info("PHASE 3: BERT Embeddings → PCA → SVM...")
        logger.info("=" * 60)

        bert_embeddings = np.load(bert_path)
        np.save(os.path.join(results_dir, 'features', 'bert_embeddings.npy'), bert_embeddings)
        logger.info(f"Loaded BERT embeddings: {bert_embeddings.shape}")

        (bert_metrics, bert_cat_df, _, _, _, _,
         _, _, _, _, _) = run_pca_svm_experiment(
            bert_embeddings, y, categories, "BERT", results_dir, logger
        )
        with open(os.path.join(results_dir, 'metrics', 'bert_global.json'), 'w') as f:
            json.dump(bert_metrics, f, indent=2)
        bert_cat_df.to_csv(os.path.join(results_dir, 'metrics', 'bert_category.csv'), index=False)

        # BERT Full Features (No PCA)
        logger.info("=" * 60)
        logger.info("PHASE 3b: BERT Embeddings → SVM (No PCA)...")
        logger.info("=" * 60)

        bert_full_metrics, bert_full_cat_df = run_svm_no_pca_experiment(
            bert_embeddings, y, categories, "bert_full", results_dir, logger
        )

        # Combined HC + BERT
        logger.info("=" * 60)
        logger.info("PHASE 4: HC + BERT Combined → PCA → SVM...")
        logger.info("=" * 60)

        X_combined = np.hstack([df_features.values, bert_embeddings])
        np.save(os.path.join(results_dir, 'features', 'combined_features.npy'), X_combined)

        (combined_metrics, combined_cat_df, _, _, _, _,
         _, _, _, _, _) = run_pca_svm_experiment(
            X_combined, y, categories, "combined", results_dir, logger
        )
        with open(os.path.join(results_dir, 'metrics', 'combined_global.json'), 'w') as f:
            json.dump(combined_metrics, f, indent=2)
        combined_cat_df.to_csv(os.path.join(results_dir, 'metrics', 'combined_category.csv'), index=False)
    else:
        logger.warning(f"No cached BERT embeddings found at {bert_path}. Skipping Phases 3, 3b & 4.")

    # ══════════════════════════════════════════════════════════
    # PHASE 5 & 6: Visualizations + Summary
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 5: Generating visualizations and comparison...")
    logger.info("=" * 60)

    comparison_rows = []
    for method, metrics in [("HC Features \u2192 PCA \u2192 SVM", hc_metrics),
                            ("HC Features \u2192 SVM", full_feat_metrics),
                            ("BERT Embeddings \u2192 PCA \u2192 SVM", bert_metrics),
                            ("BERT Embeddings \u2192 SVM", bert_full_metrics),
                            ("HC + BERT Combined \u2192 PCA \u2192 SVM", combined_metrics)]:
        if metrics is not None:
            comparison_rows.append({
                'Method': method, 'Accuracy': metrics['Accuracy'],
                'Human_F1': metrics['Human_F1'], 'LLM_F1': metrics['LLM_F1'],
                'Human_Precision': metrics['Human_Precision'], 'LLM_Precision': metrics['LLM_Precision'],
                'Human_Recall': metrics['Human_Recall'], 'LLM_Recall': metrics['LLM_Recall'],
            })

    comparison_df = pd.DataFrame(comparison_rows) if comparison_rows else None
    if comparison_df is not None:
        with open(os.path.join(results_dir, 'metrics', 'comparison_summary.json'), 'w') as f:
            json.dump(comparison_rows, f, indent=2)
        comparison_df.to_csv(os.path.join(results_dir, 'metrics', 'comparison_summary.csv'), index=False)

    generate_all_plots(
        results_dir, X_pca_hc, y, categories, pca_hc, feature_names,
        hc_cat_df, X_test_hc, y_test_hc, cat_test_hc, comparison_df, logger
    )

    logger.info("=" * 60)
    logger.info("PHASE 6: Generating final summary...")
    logger.info("=" * 60)

    summary_text = generate_summary(
        results_dir, dataset_stats, feature_names, pca_hc, ev_hc,
        hc_metrics, full_feat_metrics, bert_metrics, bert_full_metrics,
        combined_metrics, hc_cat_df, bert_cat_df, combined_cat_df, logger
    )

    print("\n" + summary_text)
    logger.info("AutoRev Pipeline (Resume Mode) Complete!")


# ═══════════════════════════════════════════════════════════════
# FULL EXECUTION PIPELINE
# ═══════════════════════════════════════════════════════════════
def main():
    # ── Setup ────────────────────────────────────────────────
    results_dir = setup_results_dir()
    logger = setup_logging(results_dir)

    logger.info("AutoRev Pipeline Starting...")
    logger.info(f"Results will be saved to: {os.path.abspath(results_dir)}")

    # ── Model Downloads ──────────────────────────────────────
    logger.info("Verifying model downloads...")
    stanza.download('mr', processors='tokenize,pos,lemma,depparse')
    _ = BPEmb(lang="mr", vs=10000)

    # ── Dataset Loading ──────────────────────────────────────
    logger.info("Loading datasets...")
    df_human = pd.read_csv(HUMAN_CSV)
    df_llm = pd.read_csv(LLM_CSV)

    df_human['Source'] = 0
    df_llm['Source'] = 1

    # Use 'Generated_Article' column for LLM text if present
    if 'Generated_Article' in df_llm.columns:
        df_llm_texts = df_llm[['Generated_Article', 'Source', 'Label']].rename(
            columns={'Generated_Article': 'Text'}
        )
    else:
        df_llm_texts = df_llm[['Text', 'Source', 'Label']]

    df_combined = pd.concat([df_human[['Text', 'Source', 'Label']], df_llm_texts], ignore_index=True)
    df_combined = df_combined.dropna(subset=['Text'])
    df_combined = df_combined[df_combined['Label'] != 'Label']  # Remove header duplication artifacts

    dataset_stats = {
        'Human articles': len(df_human),
        'LLM articles': len(df_llm),
        'Combined (after dropna)': len(df_combined),
        'Categories': sorted(df_combined['Label'].dropna().unique().tolist()),
    }
    logger.info(f"Dataset: {dataset_stats['Human articles']} human + {dataset_stats['LLM articles']} LLM = {dataset_stats['Combined (after dropna)']} total")

    # ── Reference Vocabulary ─────────────────────────────────
    top_100, top_1000 = compute_reference_vocab(df_human)

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Handcrafted Feature Extraction
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 1: Extracting 18 handcrafted features...")
    logger.info("=" * 60)

    rows_to_process = df_combined[['Text', 'Source', 'Label']].to_dict('records')
    num_cores = 2
    logger.info(f"Spinning up pool with {num_cores} CPU cores...")

    feature_list, valid_labels, valid_categories, valid_texts = [], [], [], []

    with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(top_100, top_1000)) as pool:
        results = list(tqdm(
            pool.imap(process_single_row, rows_to_process),
            total=len(rows_to_process), desc="Feature extraction"
        ))

    for idx, (feats, label, category) in enumerate(results):
        if feats is not None:
            feature_list.append(feats)
            valid_labels.append(label)
            valid_categories.append(category)
            valid_texts.append(str(rows_to_process[idx]['Text']).strip())

    df_features = pd.DataFrame(feature_list)
    y = np.array(valid_labels)
    categories = np.array(valid_categories)
    feature_names = df_features.columns.tolist()

    logger.info(f"Features extracted for {len(df_features)} texts ({len(feature_names)} features)")

    # Save raw features
    df_features_out = df_features.copy()
    df_features_out['label'] = y
    df_features_out['category'] = categories
    df_features_out.to_csv(
        os.path.join(results_dir, "features", "handcrafted_features.csv"), index=False
    )
    df_features_out.to_csv("extracted_features.csv", index=False)  # Also save to project root
    logger.info("Saved: features/handcrafted_features.csv + extracted_features.csv (project root)")

    # ══════════════════════════════════════════════════════════
    # PHASE 2: PCA + SVM on Handcrafted Features
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2: PCA + SVM on handcrafted features...")
    logger.info("=" * 60)

    (hc_metrics, hc_cat_df, X_pca_hc, pca_hc, svm_hc, scaler_hc,
     X_train_hc, X_test_hc, y_train_hc, y_test_hc, cat_test_hc) = run_pca_svm_experiment(
        df_features.values, y, categories, "Handcrafted", results_dir, logger, save_models=True
    )

    # Save PCA loadings
    ev_hc = pca_hc.explained_variance_ratio_
    loadings_df = pd.DataFrame(
        pca_hc.components_.T, index=feature_names, columns=['PC1', 'PC2']
    )
    loadings_df.to_csv(os.path.join(results_dir, "pca", "pca_loadings.csv"))

    ev_dict = {
        'PC1_variance_ratio': float(ev_hc[0]),
        'PC2_variance_ratio': float(ev_hc[1]),
        'total_variance_ratio': float(sum(ev_hc)),
    }
    with open(os.path.join(results_dir, "pca", "explained_variance.json"), "w") as f:
        json.dump(ev_dict, f, indent=2)

    pca_coords_df = pd.DataFrame(X_pca_hc, columns=['PC1', 'PC2'])
    pca_coords_df['label'] = y
    pca_coords_df['category'] = categories
    pca_coords_df.to_csv(os.path.join(results_dir, "pca", "pca_transformed.csv"), index=False)

    # Save metrics
    with open(os.path.join(results_dir, "metrics", "handcrafted_global.json"), "w") as f:
        json.dump(hc_metrics, f, indent=2)
    hc_cat_df.to_csv(os.path.join(results_dir, "metrics", "handcrafted_category.csv"), index=False)

    logger.info("Saved: pca/, metrics/ for handcrafted features")

    # ══════════════════════════════════════════════════════════
    # PHASE 2b: Full Features SVM (No PCA)
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2b: SVM on full feature set (no PCA)...")
    logger.info("=" * 60)

    full_feat_metrics, full_feat_cat_df = run_svm_no_pca_experiment(
        df_features.values, y, categories, "full_features", results_dir, logger
    )

    # ══════════════════════════════════════════════════════════
    # PHASE 3: BERT Baseline
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"PHASE 3: BERT Embeddings \u2192 PCA \u2192 SVM ({BERT_MODEL_NAME})...")
    logger.info("=" * 60)

    bert_metrics = None
    bert_cat_df = None
    bert_full_metrics = None
    bert_full_cat_df = None

    try:
        bert_embeddings = extract_bert_embeddings(valid_texts, BERT_MODEL_NAME)
        np.save(os.path.join(results_dir, "features", "bert_embeddings.npy"), bert_embeddings)
        logger.info(f"BERT embeddings shape: {bert_embeddings.shape}")

        (bert_metrics, bert_cat_df, X_pca_bert, pca_bert, svm_bert, scaler_bert,
         _, X_test_bert, _, y_test_bert, cat_test_bert) = run_pca_svm_experiment(
            bert_embeddings, y, categories, "BERT", results_dir, logger
        )

        with open(os.path.join(results_dir, "metrics", "bert_global.json"), "w") as f:
            json.dump(bert_metrics, f, indent=2)
        bert_cat_df.to_csv(os.path.join(results_dir, "metrics", "bert_category.csv"), index=False)
        logger.info("Saved: metrics/ for BERT PCA+SVM")

        # BERT Full Features (No PCA)
        logger.info("=" * 60)
        logger.info("PHASE 3b: BERT Embeddings \u2192 SVM (No PCA)...")
        logger.info("=" * 60)

        bert_full_metrics, bert_full_cat_df = run_svm_no_pca_experiment(
            bert_embeddings, y, categories, "bert_full", results_dir, logger
        )

    except Exception as e:
        logger.error(f"BERT experiments failed: {e}")
        logger.error("Continuing without BERT results...")

    # ══════════════════════════════════════════════════════════
    # PHASE 4: HC + BERT Combined \u2192 PCA \u2192 SVM
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 4: HC + BERT Combined \u2192 PCA \u2192 SVM...")
    logger.info("=" * 60)

    combined_metrics = None
    combined_cat_df = None

    if bert_metrics is not None:
        try:
            X_combined = np.hstack([df_features.values, bert_embeddings])
            np.save(os.path.join(results_dir, "features", "combined_features.npy"), X_combined)
            logger.info(f"Combined features shape: {X_combined.shape}")

            (combined_metrics, combined_cat_df, _, _, _, _,
             _, _, _, _, _) = run_pca_svm_experiment(
                X_combined, y, categories, "combined", results_dir, logger
            )

            with open(os.path.join(results_dir, "metrics", "combined_global.json"), "w") as f:
                json.dump(combined_metrics, f, indent=2)
            combined_cat_df.to_csv(
                os.path.join(results_dir, "metrics", "combined_category.csv"), index=False
            )
            logger.info("Saved: metrics/ for combined study")

        except Exception as e:
            logger.error(f"Combined study failed: {e}")
    else:
        logger.warning("Skipping combined study \u2014 BERT was unavailable.")

    # ══════════════════════════════════════════════════════════
    # PHASE 5: Comparison & Visualization
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 5: Generating visualizations and comparison...")
    logger.info("=" * 60)

    # Build comparison dataframe
    comparison_rows = []
    for method, metrics in [("HC Features \u2192 PCA \u2192 SVM", hc_metrics),
                            ("HC Features \u2192 SVM", full_feat_metrics),
                            ("BERT Embeddings \u2192 PCA \u2192 SVM", bert_metrics),
                            ("BERT Embeddings \u2192 SVM", bert_full_metrics),
                            ("HC + BERT Combined \u2192 PCA \u2192 SVM", combined_metrics)]:
        if metrics is not None:
            comparison_rows.append({
                'Method': method,
                'Accuracy': metrics['Accuracy'],
                'Human_F1': metrics['Human_F1'],
                'LLM_F1': metrics['LLM_F1'],
                'Human_Precision': metrics['Human_Precision'],
                'LLM_Precision': metrics['LLM_Precision'],
                'Human_Recall': metrics['Human_Recall'],
                'LLM_Recall': metrics['LLM_Recall'],
            })

    comparison_df = pd.DataFrame(comparison_rows) if comparison_rows else None

    if comparison_df is not None:
        with open(os.path.join(results_dir, "metrics", "comparison_summary.json"), "w") as f:
            json.dump(comparison_rows, f, indent=2)
        comparison_df.to_csv(
            os.path.join(results_dir, "metrics", "comparison_summary.csv"), index=False
        )

    generate_all_plots(
        results_dir, X_pca_hc, y, categories, pca_hc, feature_names,
        hc_cat_df, X_test_hc, y_test_hc, cat_test_hc, comparison_df, logger
    )

    # ══════════════════════════════════════════════════════════
    # PHASE 6: Summary
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 6: Generating final summary...")
    logger.info("=" * 60)

    summary_text = generate_summary(
        results_dir, dataset_stats, feature_names, pca_hc, ev_hc,
        hc_metrics, full_feat_metrics, bert_metrics, bert_full_metrics,
        combined_metrics, hc_cat_df, bert_cat_df, combined_cat_df, logger
    )

    print("\n" + summary_text)
    logger.info("AutoRev Pipeline Complete!")


if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="AutoRev: Marathi Text Detection Pipeline")
    parser.add_argument('--from-run', type=str, default=None,
                        help='Path to a previous run directory to resume from (skips feature extraction)')
    args = parser.parse_args()

    if args.from_run:
        main_from_run(args.from_run)
    else:
        main()