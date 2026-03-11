# +
import umap
import evaluate
import nltk
import spacy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams

import torch
from bert_score import BERTScorer


# -

# # Fidelity evaluation

# ## Maximum Mean Discrepancy (MMD)

def calculate_mmd(X, Y, gamma=1.0):
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


# ## BERTScore

def get_bertscore(real_reports, da_reports):
    scorer = BERTScorer(lang="es", model_type="bert-base-multilingual-cased", rescale_with_baseline=False)

    da_reports_clean = [
        str(report) for report in da_reports 
        if str(report).strip() != "" and str(report).strip().lower() != "nan"
    ]
    real_reports_clean = [
        str(report) for report in real_reports 
        if str(report).strip() != "" and str(report).strip().lower() != "nan"
    ]
    random_sample = np.random.choice(real_reports_clean, len(da_reports_clean)).tolist()

    return scorer.score(da_reports_clean, random_sample)


# ## N-Grams

def get_top_ngrams(corpus, n=2, top_k=10):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]


# ## ROUGE

def get_rouge(real_reports, da_reports):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=da_reports,
                         references= np.random.choice(real_reports, len(da_reports)))


# ## METEOR

# +
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

def get_meteor(real_reports, da_reports):
    meteor = evaluate.load("meteor")
    return meteor.compute(predictions=da_reports, 
                                    references= np.random.choice(real_reports, len(da_reports)))


# -

# ## Sentence Mover's Similarity (SMS)

# +
nlp = spacy.load("es_core_news_md")

def calculate_sms(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

def calculate_all_sms(real_reports, da_reports):
    return [calculate_sms(h, s) for h, s in zip(np.random.choice(real_reports, len(da_reports)), da_reports)]


# -

# ## Embedding visualization

def show_embeddings(emb_real, emb_da):
    all_embeddings = np.vstack([emb_real, emb_da])
    labels = ['Real'] * len(emb_real) + ['Synthetic'] * len(emb_da)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    embeddings_2d_tsne = tsne.fit_transform(all_embeddings)
    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, verbose=True)
    embeddings_2d_umap = reducer.fit_transform(all_embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=embeddings_2d_tsne[:,0], y=embeddings_2d_tsne[:,1], hue=labels, ax=ax1, alpha=0.6)
    ax1.set_title('t-SNE embedding map')
    sns.scatterplot(x=embeddings_2d_umap[:,0], y=embeddings_2d_umap[:,1], hue=labels, ax=ax2, alpha=0.6)
    ax2.set_title('UMAP embedding map')
    plt.tight_layout()
    plt.show()


# # Diversity evaluation

# ## Self-BLEU

def calculate_self_bleu(sentences, n_samples=100):
    if len(sentences) > n_samples:
        indices = np.random.choice(len(sentences), n_samples, replace=False)
        sample = [sentences[i].split() for i in indices]
    else:
        sample = [s.split() for s in sentences]
    
    scores = []
    smoothing = SmoothingFunction().method1
    for i in range(len(sample)):
        hypothesis = sample[i]
        references = sample[:i] + sample[i+1:]
        score = sentence_bleu(references, hypothesis, smoothing_function=smoothing)
        scores.append(score)
    return np.mean(scores)


# ## Type-Token Ratio (TTR)

def lexical_diversity(sentences):
    ttrs = []
    for s in sentences:
        words = s.lower().split()
        if len(words) == 0: continue
        ttr = len(set(words)) / len(words)
        ttrs.append(ttr)
    return np.mean(ttrs)


# ## Nearest Neighbor Distance (NND)

def calculate_nnd(emb_real, emb_da):
    dist_matrix = pairwise_distances(emb_da, emb_real, metric='cosine')
    min_distances = np.min(dist_matrix, axis=1)
    return np.mean(min_distances), min_distances


def get_plagiarism_candidates(synth_texts, real_texts, synth_embs, real_embs, threshold=0.05):
    dist_matrix = cosine_distances(synth_embs, real_embs)
    min_distances = np.min(dist_matrix, axis=1)
    closest_real_idx = np.argmin(dist_matrix, axis=1)
    
    suspect_indices = np.where(min_distances < threshold)[0]
    
    results = []
    for idx in suspect_indices:
        results.append({
            "Synthetic_ID": idx,
            "NND": round(min_distances[idx], 4),
            "Synthetic report": synth_texts[idx],
            "Closest neighbor": real_texts[closest_real_idx[idx]]
        })
    
    return pd.DataFrame(results).sort_values(by="NND")
