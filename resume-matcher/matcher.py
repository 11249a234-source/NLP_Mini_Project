"""Semantic resume matching and explicit JD skill-gap analysis."""

import re
from typing import Any

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "all-MiniLM-L6-v2"


def load_sbert_model() -> SentenceTransformer:
    """Load the embedding model; callers should cache this resource."""
    return SentenceTransformer(MODEL_NAME)


def compute_similarity(jd_text: str, resume_text: str, model: Any) -> float:
    """Return cosine similarity as a non-negative percentage in [0, 100]."""
    if not jd_text or not jd_text.strip() or not resume_text or not resume_text.strip():
        return 0.0

    embeddings = model.encode(
        [jd_text.strip(), resume_text.strip()],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    cosine_score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
    return round(max(0.0, min(1.0, cosine_score)) * 100, 2)


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _skill_is_present(skill: str, candidate_skill: str) -> bool:
    if skill == candidate_skill:
        return True
    if len(skill) == 1:
        return skill in candidate_skill.split()

    pattern = rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])"
    return re.search(pattern, candidate_skill) is not None


def analyze_skill_gap(jd_keywords: list[str], extracted_skills: list[str]) -> dict[str, list[str]]:
    """Identify matched and missing JD terms using case-insensitive phrase boundaries."""
    unique_jd_skills = {}
    for skill in jd_keywords:
        normalized = _normalize(skill)
        if normalized:
            unique_jd_skills.setdefault(normalized, skill.strip())

    candidate_skills = [_normalize(skill) for skill in extracted_skills if _normalize(skill)]
    matched = []
    missing = []

    for normalized, original in unique_jd_skills.items():
        if any(_skill_is_present(normalized, candidate) for candidate in candidate_skills):
            matched.append(original)
        else:
            missing.append(original)

    return {"matched": matched, "missing": missing}