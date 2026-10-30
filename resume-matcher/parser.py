"""PDF text extraction and resume entity parsing."""

from pathlib import Path
from typing import BinaryIO

import pdfplumber
import spacy
from spacy.language import Language


MODEL_PATH = Path(__file__).resolve().parent / "models" / "spacy_ner"
ENTITY_LABELS = ("SKILL", "PROJECT", "DEGREE", "CERTIFICATION", "EXPERIENCE")

FALLBACK_ENTITIES = {
    "SKILL": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "SQL", "R",
        "Go", "Rust", "HTML", "CSS", "React", "Angular", "Vue", "Node.js",
        "Django", "Flask", "FastAPI", "Spring", "REST API", "GraphQL", "Git",
        "GitHub", "Docker", "Kubernetes", "AWS", "Azure", "Google Cloud",
        "Linux", "CI/CD", "Machine Learning", "Deep Learning", "Natural Language Processing",
        "NLP", "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy",
        "Data Analysis", "Data Visualization", "Power BI", "Tableau", "Excel",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Apache Spark", "Hadoop",
        "Communication", "Leadership", "Project Management", "Agile", "Scrum",
    ],
    "DEGREE": [
        "Bachelor of Science", "Bachelor of Arts", "Bachelor of Engineering",
        "Master of Science", "Master of Arts", "Master of Business Administration",
        "Bachelor's degree", "Master's degree", "Associate degree", "Ph.D.",
        "B.S.", "B.A.", "B.Eng.", "M.S.", "M.A.", "M.B.A.", "B.Sc.", "M.Sc.",
    ],
    "CERTIFICATION": [
        "AWS Certified Solutions Architect", "AWS Certified Developer",
        "Microsoft Certified: Azure Fundamentals", "Google Cloud Professional",
        "Certified Kubernetes Administrator", "CompTIA Security+", "CompTIA A+",
        "PMP", "Project Management Professional", "Certified ScrumMaster",
        "TensorFlow Developer Certificate", "Cisco Certified Network Associate", "CCNA",
    ],
    "PROJECT": [],
    "EXPERIENCE": [],
}


def _build_pipeline() -> tuple[Language, bool]:
    """Load the trained pipeline, or use phrase rules for a useful local fallback."""
    config_path = MODEL_PATH / "config.cfg"
    if config_path.is_file():
        try:
            return spacy.load(str(MODEL_PATH)), True
        except (OSError, IOError) as exc:
            raise RuntimeError(f"Could not load the SpaCy model at {MODEL_PATH}: {exc}") from exc

    pipeline = spacy.blank("en")
    ruler = pipeline.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    ruler.add_patterns(
        [
            {"label": label, "pattern": phrase}
            for label, phrases in FALLBACK_ENTITIES.items()
            for phrase in phrases
        ]
    )
    return pipeline, False


nlp, USING_CUSTOM_MODEL = _build_pipeline()


def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    """Extract selectable text from each page, preserving page order."""
    if hasattr(pdf_file, "seek"):
        pdf_file.seek(0)

    page_text = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True) or page.extract_text()
            if text:
                page_text.append(text.strip())

    if hasattr(pdf_file, "seek"):
        pdf_file.seek(0)
    return "\n\n".join(page_text)


def parse_resume_entities(text: str) -> dict[str, list[str]]:
    """Extract recognized resume entities, deduplicating case-insensitively."""
    entities = {label: [] for label in ENTITY_LABELS}
    seen = {label: set() for label in ENTITY_LABELS}

    if not text or not text.strip():
        return entities

    for ent in nlp(text).ents:
        label = ent.label_
        value = ent.text.strip()
        normalized = value.casefold()
        if label in entities and value and normalized not in seen[label]:
            entities[label].append(value)
            seen[label].add(normalized)

    return entities