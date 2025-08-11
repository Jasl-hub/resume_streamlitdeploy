# AI-Powered Resume Shortlisting Tool

An AI-driven resume shortlisting system built with Python, Streamlit, and Transformer-based NLP models.
This tool ranks resumes against a given Job Description (JD) using a Hybrid Scoring Method that combines **TF-IDF + SpaCy embeddings** and **BERT (SBERT) similarity**, along with an **Experience Bonus**.

---

## 🚀 Features
- **Hybrid Scoring** → TF-IDF + SpaCy (semantic) + SBERT (contextual)
- **Experience Bonus** → Adds extra weight for candidates with more years of experience
- **Multi-format Input** → Supports PDF resumes
- **Real-time Ranking** → Streamlit UI for instant results
- **Deployable** → Easily hostable on Streamlit Cloud or Docker

---

## 🏗 Architecture

```
flowchart LR
    A[Job Description (PDF/Text)] --> B[Preprocessing & Cleaning]
    B --> C[TF-IDF + SpaCy Embeddings]
    B --> D[SBERT Embeddings]
    C --> E[Hybrid Scoring Engine]
    D --> E
    E --> F[Ranked Candidate List]
    F --> G[Streamlit Frontend (main.py)]
```


---

## 📂 Project Structure

```
resume_streamlitdeploy/
│── main.py              # Streamlit frontend
│── model.py             # Core AI models and scoring logic
│── utils.py             # Helper functions (PDF reading, text extraction, etc.)
│── requirements.txt     # Dependencies
│── Dockerfile           # For containerized deployment
│── .streamlit/          # Streamlit configuration
```

---

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/resume_streamlitdeploy.git
cd resume_streamlitdeploy
```

### 2️⃣ Create a Virtual Environment

**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Install SpaCy Models
```bash
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
```

---

## ▶ Running Locally
```bash
streamlit run main.py
```
Then open the link in your terminal (e.g., `http://localhost:8501`).

---

## ☁ Deploy to Streamlit Cloud
- Push your code to GitHub.
- Go to **Streamlit Cloud**, create a new app, and select your repository.
- Add the following to **requirements.txt** so models are preinstalled:

```
https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Deploy 🚀

---

## 📊 Scoring Method
- **TF-IDF + SpaCy Similarity** → Combines keyword overlap with semantic similarity.
- **SBERT (Sentence-BERT)** → Captures contextual meaning.
- **Hybrid Score** → Weighted average of both methods + experience bonus.
- **Experience Bonus** → Extra points for ≥3 years experience.

---

## 📌 Example Output
```
🏆 Top Candidates
📌 candidate_069.pdf — Score: 0.7708
📌 candidate_053.pdf — Score: 0.7690
📌 candidate_099.pdf — Score: 0.7666
📌 candidate_017.pdf — Score: 0.7665
📌 candidate_135.pdf — Score: 0.7635
```

---

## 📌 Future Improvements
- Add skill-based keyword extraction and matching
- Add `.docx` resume parsing fallback (already partially supported via docx2txt)
- Allow adjustable weights for TF-IDF and BERT in UI sliders
- Add a per-candidate breakdown showing TF-IDF score, SpaCy score, BERT score, and experience bonus

---

## 📜 License
This project is licensed under the MIT License.

---

💡 **Pro Tip for Recruiters:** This tool can be easily adapted to include skills extraction, certifications scoring, or domain-specific filtering.
