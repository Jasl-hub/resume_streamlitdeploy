import os
import re
import numpy as np
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from datetime import datetime
from scipy.stats import spearmanr
import spacy
from spacy.cli import download

# ✅ Lazy-load spaCy models at runtime
def load_spacy_models():
    try:
        spacy_lg = spacy.load("en_core_web_md")
    except OSError:
        download("en_core_web_md")
        spacy_lg = spacy.load("en_core_web_md")

    try:
        spacy_sm = spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        spacy_sm = spacy.load("en_core_web_sm")

    return spacy_lg, spacy_sm


# Hugging Face tokenizer/model can be global (safe to load during build)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


def extract_years_of_experience(text):
    total_years = 0.0
    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s*)?experience', re.IGNORECASE)
    for match in pattern.findall(text):
        try:
            total_years += float(match)
        except:
            continue
    return total_years


def compute_experience_bonus(text):
    years = extract_years_of_experience(text)
    if years >= 5:
        return 1.0
    elif years >= 3:
        return 0.5
    return 0.0


def get_spacy_embedding(text):
    spacy_lg, _ = load_spacy_models()
    return spacy_lg(text).vector


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def tfidf_score(text1, text2):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def spacy_tfidf_combined_score(resume_text, jd_text):
    tfidf_sim = tfidf_score(resume_text, jd_text)
    spacy_sim = cosine_similarity([get_spacy_embedding(resume_text)], [get_spacy_embedding(jd_text)])[0][0]
    return (tfidf_sim + spacy_sim) / 2


def bert_score(resume_text, jd_text):
    return cosine_similarity([get_bert_embedding(resume_text)], [get_bert_embedding(jd_text)])[0][0]


def evaluate_resumes(jd_path, resumes_folder):
    jd_text = extract_text_from_pdf(jd_path)
    tfidf_spacy_scores = []
    bert_scores = []

    for resume_file in sorted(os.listdir(resumes_folder)):
        resume_path = os.path.join(resumes_folder, resume_file)
        resume_text = extract_text_from_pdf(resume_path)
        experience_bonus = compute_experience_bonus(resume_text)

        tfidf_spacy = spacy_tfidf_combined_score(resume_text, jd_text) + experience_bonus
        bert = bert_score(resume_text, jd_text) + experience_bonus

        tfidf_spacy_scores.append((resume_file, tfidf_spacy))
        bert_scores.append((resume_file, bert))

    return sorted(tfidf_spacy_scores, key=lambda x: x[1], reverse=True), sorted(bert_scores, key=lambda x: x[1], reverse=True)


def compare_results(tfidf_spacy_top, bert_top, k=15):
    tfidf_top_k = [x[0] for x in tfidf_spacy_top[:k]]
    bert_top_k = [x[0] for x in bert_top[:k]]

    print("\nTop K Candidates (TF-IDF + SpaCy):")
    for name, score in tfidf_spacy_top[:k]:
        print(f"{name}: {score:.4f}")

    print("\nTop K Candidates (BERT):")
    for name, score in bert_top[:k]:
        print(f"{name}: {score:.4f}")

    overlap = set(tfidf_top_k).intersection(set(bert_top_k))
    print(f"\n🔁 Overlap in Top-{k}: {len(overlap)} resumes: {overlap}")

    common_candidates = [name for name in tfidf_top_k if name in bert_top_k]
    if common_candidates:
        tfidf_ranks = [i for i, (name, _) in enumerate(tfidf_spacy_top) if name in common_candidates]
        bert_ranks = [i for i, (name, _) in enumerate(bert_top) if name in common_candidates]
        corr, _ = spearmanr(tfidf_ranks, bert_ranks)
        print(f"📊 Spearman Rank Correlation: {corr:.4f}")
    else:
        print("No common candidates to compute rank correlation.")


def evaluate_hybrid_scores(jd_path, resumes_folder, tfidf_weight=0.5, bert_weight=0.5):
    jd_text = extract_text_from_pdf(jd_path)
    hybrid_scores = []
    tfidf_spacy_scores = []
    bert_scores = []

    for resume_file in sorted(os.listdir(resumes_folder)):
        resume_path = os.path.join(resumes_folder, resume_file)
        resume_text = extract_text_from_pdf(resume_path)
        experience_bonus = compute_experience_bonus(resume_text)

        tfidf_spacy = spacy_tfidf_combined_score(resume_text, jd_text)
        bert = bert_score(resume_text, jd_text)
        hybrid = (tfidf_weight * tfidf_spacy + bert_weight * bert) + experience_bonus

        hybrid_scores.append((resume_file, hybrid))
        tfidf_spacy_scores.append((resume_file, tfidf_spacy + experience_bonus))
        bert_scores.append((resume_file, bert + experience_bonus))

    return (
        sorted(tfidf_spacy_scores, key=lambda x: x[1], reverse=True),
        sorted(bert_scores, key=lambda x: x[1], reverse=True),
        sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    )


def evaluate_hybrid_scores_from_text(jd_text, resume_dict, tfidf_weight=0.5, bert_weight=0.5):
    hybrid_scores = {}
    for filename, resume_text in resume_dict.items():
        experience_bonus = compute_experience_bonus(resume_text)
        tfidf_spacy = spacy_tfidf_combined_score(resume_text, jd_text)
        bert = bert_score(resume_text, jd_text)
        hybrid = (tfidf_weight * tfidf_spacy + bert_weight * bert) + experience_bonus
        hybrid_scores[filename] = round(hybrid, 4)

    return dict(sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True))
