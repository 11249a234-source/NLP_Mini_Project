# Automated Resume Matcher & Skill Gap Analyzer

A local Streamlit app that extracts resume text from PDFs, estimates semantic similarity to a job description, and reports matched and missing skills.

## Run locally

Python 3.10 or newer is recommended.

```bash
cd resume-matcher
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.
The app opens at `http://localhost:8501`. The first run downloads the
`all-MiniLM-L6-v2` embedding model, so have an internet connection available
before presenting. Later runs use the local model cache.

## SpaCy extraction

Place a trained SpaCy pipeline in `models/spacy_ner/`. It should include a
`config.cfg` and emit entities labeled `SKILL`, `PROJECT`, `DEGREE`,
`CERTIFICATION`, or `EXPERIENCE` to populate those fields.

When no trained model is installed, the app uses built-in phrase rules for
common skills, degrees, and certifications. It does not infer project names or
work experience in fallback mode. The phrase rules are a demo-friendly
baseline, not a substitute for a domain-trained NER model.

## Using the app

1. Paste a job description in the sidebar and optionally enter target skills.
2. Upload one or more text-based PDF resumes.
3. Select **Analyze resumes** to view the leaderboard and candidate details.

If the skill field is empty, the app tries to identify target skills in the job
description. Image-only PDFs are not OCR-processed. Resumes are processed by
the local app and are not uploaded to a resume-matching service; the embedding
weights may be downloaded from Hugging Face on first use.

The semantic score is an approximate relevance signal, not a hiring decision
or a validated measure of candidate ability. Review the underlying resume
content and skill matches rather than relying on the ranking alone.

For architecture, algorithms, data structures, and limitations, see the
[technical documentation](TECHNICAL_README.md).