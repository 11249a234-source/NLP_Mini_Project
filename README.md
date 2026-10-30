# Technical Documentation: Automated Resume Matcher & Skill Gap Analyzer

This document describes the system architecture, component responsibilities,
algorithm workflow, data structures, trade-offs, and extension points.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Specifications](#component-specifications)
3. [Algorithmic Workflow](#algorithmic-workflow)
4. [Data Flow and Structures](#data-flow-and-structures)
5. [Performance and Trade-offs](#performance-and-trade-offs)
6. [Future Enhancements](#future-enhancements)

## 1. System Architecture

The app has three runtime modules, with model assets kept separately:

```text
                  +------------------------+
                  |   Streamlit Web UI     |
                  |        (app.py)        |
                  +-----------+------------+
                              |
                   +----------+----------+
                   |                     |
          +--------v---------+  +--------v---------+
          |  Parser Module   |  |  Matcher Module  |
          |   (parser.py)    |  |   (matcher.py)   |
          +--------+---------+  +--------+---------+
                   |                     |
           +-------+-------+     +-------+--------+
           |  pdfplumber   |     | Sentence-      |
           |  and SpaCy    |     | Transformers    |
           +---------------+     +----------------+
                   |                     |
           models/spacy_ner/      all-MiniLM-L6-v2
```

- **UI layer (`app.py`):** collects the JD, target skills, and PDF uploads;
  coordinates analysis; caches the embedding model; and presents rankings,
  candidate details, and CSV export.
- **Extraction engine (`parser.py`):** extracts text page by page and parses
  entities with either a custom SpaCy pipeline or built-in phrase rules.
- **NLP and scoring engine (`matcher.py`):** computes semantic similarity and
  compares normalized target skill phrases with extracted skill phrases.
- **Model assets:** the custom SpaCy pipeline is optional and belongs in
  `models/spacy_ner/`; the Sentence-Transformer model is fetched from the model
  hub on first load and cached locally by its library.

## 2. Component Specifications

### 2.1 PDF Parsing (`pdfplumber`)

`extract_text_from_pdf` opens the uploaded PDF and extracts each page with
`layout=True`, falling back to the default text extraction if needed. Non-empty
page text is concatenated in page order. Layout-aware extraction can help retain
spacing and columns, but PDF reading order varies; complex or image-only PDFs
may still need manual review or OCR.

### 2.2 Named Entity Recognition (SpaCy)

The optional custom model path is `models/spacy_ner/`. The parser accepts these
labels:

- `SKILL`: tools, frameworks, and domain knowledge.
- `DEGREE`: educational qualifications.
- `CERTIFICATION`: professional credentials.
- `PROJECT`: named projects or initiatives.
- `EXPERIENCE`: work history or professional roles.

When a model with `config.cfg` is present, the app loads it with `spacy.load`.
The custom pipeline must emit the labels above for those fields to appear.

Without a custom model, the app builds a blank English pipeline with SpaCy's
`EntityRuler` and a bundled phrase list. The fallback covers common skills,
degrees, and certifications; it does not infer project names or work history.
It is a transparent baseline, not a trained NER model, and its vocabulary is
limited to the phrases in `parser.py`.

### 2.3 Semantic Embedding and Scoring (`Sentence-Transformers`)

- **Model:** `all-MiniLM-L6-v2`.
- **Embedding size:** 384 dimensions.
- **Loading:** `app.py` caches the model with `st.cache_resource`.
- **Metric:** cosine similarity between normalized JD and resume embeddings.

For embeddings $\mathbf{u}$ and $\mathbf{v}$, cosine similarity is:

$$
\operatorname{cos}(\mathbf{u}, \mathbf{v}) =
\frac{\mathbf{u} \cdot \mathbf{v}}
{\lVert\mathbf{u}\rVert_2\,\lVert\mathbf{v}\rVert_2}
$$

The app converts this value to a percentage and clamps it to $[0, 100]$:

$$
\text{score} = 100 \times \max(0, \min(1, \operatorname{cos}(\mathbf{u}, \mathbf{v})))
$$

This is a relative semantic relevance signal, not a probability or a validated
measure of candidate ability. The model has a finite input length; long JD or
resume text may be truncated by the tokenizer.

### 2.4 Skill-Gap Analysis

Target skills are entered as a comma-separated list. If none are entered, the
app uses skills recognized in the JD text. Both JD and candidate phrases are
case-folded and whitespace-normalized. Phrase-boundary matching avoids common
substring mistakes (for example, matching `C` inside `C++`). Duplicate JD
skills are removed while retaining their displayed spelling.

## 3. Algorithmic Workflow

For each uploaded resume $i$:

1. **Acquire inputs:** read JD text $J$, optional target skills $K_J$, and PDF
   bytes for each resume.
2. **Extract text:**

   $$T_i = \operatorname{ExtractPDF}(\operatorname{PDF}_i)$$

3. **Extract entities:**

   $$E_i = \operatorname{ParseEntities}(T_i)$$

4. **Choose target skills:** use supplied skills, or parse skill phrases from
   the JD if the field is empty.
5. **Encode and score:** encode $J$ and $T_i$, compute cosine similarity, and
   convert it to a percentage $S_i$.
6. **Find gaps:** normalize JD skills and extracted candidate skills, then
   match each JD phrase against candidate phrases with boundaries:

   $$
   \text{Matched}_i = \{k \in K_J : \exists r \in K_{R_i},\; k \text{ matches } r\}
   $$

   $$
   \text{Missing}_i = K_J \setminus \text{Matched}_i
   $$

7. **Present results:** sort candidate records by $S_i$ descending, show the
   leaderboard and candidate details, and allow results to be downloaded as
   CSV.

Each resume is handled independently. A parsing error for one file is reported
without stopping analysis of the remaining files.

## 4. Data Flow and Structures

The parser returns a dictionary of entity labels to deduplicated string lists:

```python
{
    "SKILL": ["Python", "PyTorch", "Docker", "SQL"],
    "DEGREE": ["Bachelor of Science"],
    "CERTIFICATION": ["AWS Certified Solutions Architect"],
    "PROJECT": [],
    "EXPERIENCE": [],
}
```

The matcher returns `{"matched": [...], "missing": [...]}`. The app combines
these results into one candidate record containing the filename, score,
extracted entity lists, and skill-gap lists. Records are stored in session
state and represented as a pandas DataFrame for sorting, display, and CSV
export. Uploaded resume contents are not persisted by the application.

## 5. Performance and Trade-offs

| Choice | Advantage | Trade-off or limitation |
| --- | --- | --- |
| `pdfplumber` layout extraction | Can retain useful spacing and column layout | Extraction order can vary and parsing is CPU-bound; image-only PDFs need OCR |
| `all-MiniLM-L6-v2` | Compact embeddings and local inference after model download | Finite tokenizer input length; first use downloads model files |
| Custom SpaCy NER | Can recognize domain-specific entities | Requires labeled training data and a compatible model pipeline |
| EntityRuler fallback | Works without custom model files and is easy to inspect | Only recognizes phrases explicitly listed in `parser.py` |
| In-memory per-resume scoring | Simple and suitable for small batches | Work grows with the number and length of resumes; no vector index is used |
| Local processing | Resume contents are processed by the local app rather than a resume API | Model download requires network access on first use unless pre-cached |

## 6. Future Enhancements

1. **OCR integration:** add an OCR path for scanned or image-only PDFs.
2. **Expanded entity extraction:** train and evaluate a SpaCy NER pipeline for
   projects, experience, and domain-specific skills.
3. **Skill aliases and taxonomy:** support explicit equivalents such as
   `scikit-learn` and `sklearn`, and allow users to manage a skill vocabulary.
4. **Large-scale search:** evaluate FAISS or another vector index if the app
   needs to compare candidates across a persistent, large resume collection.
5. **Learning-oriented feedback:** provide curated learning resources for
   missing skills, while keeping matching results reviewable and advisory.
