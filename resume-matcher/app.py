"""Streamlit interface for ranking PDF resumes against a job description."""

import pandas as pd
import streamlit as st

from matcher import analyze_skill_gap, compute_similarity, load_sbert_model
from parser import USING_CUSTOM_MODEL, extract_text_from_pdf, parse_resume_entities


st.set_page_config(page_title="Resume Matcher & Skill Gap Analyzer", layout="wide")


@st.cache_resource
def get_model():
    return load_sbert_model()


def _input_signature(jd_text: str, skills: list[str], files) -> tuple:
    return (
        jd_text.strip(),
        tuple(skill.casefold() for skill in skills),
        tuple((file.name, file.size) for file in files),
    )


def _analyze_resumes(jd_text: str, jd_skills: list[str], uploaded_files, model):
    results = []
    errors = []

    target_skills = jd_skills or parse_resume_entities(jd_text)["SKILL"]
    for file in uploaded_files:
        try:
            raw_text = extract_text_from_pdf(file)
            if not raw_text.strip():
                raise ValueError("No selectable text found; the PDF may be image-only.")

            entities = parse_resume_entities(raw_text)
            gap = analyze_skill_gap(target_skills, entities["SKILL"])
            results.append(
                {
                    "Candidate": f"{len(results) + 1} - {file.name}",
                    "Filename": file.name,
                    "Match Score (%)": compute_similarity(jd_text, raw_text, model),
                    "Extracted Skills": entities["SKILL"],
                    "Matched Skills": gap["matched"],
                    "Missing Skills": gap["missing"],
                    "Degrees": entities["DEGREE"],
                    "Certifications": entities["CERTIFICATION"],
                    "Projects": entities["PROJECT"],
                    "Experience": entities["EXPERIENCE"],
                }
            )
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")

    results.sort(key=lambda result: result["Match Score (%)"], reverse=True)
    return results, errors, target_skills


st.title("Automated Resume Matcher & Skill Gap Analyzer")
st.caption("Compare resume content with a role, then review the evidence and skill gaps.")

with st.sidebar:
    st.header("Target role")
    jd_text = st.text_area("Job description", height=250, placeholder="Paste the full job description")
    jd_skills_input = st.text_input(
        "Key skills (comma-separated)",
        placeholder="Python, SQL, PyTorch, Docker",
    )
    jd_skills = [skill.strip() for skill in jd_skills_input.split(",") if skill.strip()]

st.header("Candidate resumes")
uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)

if USING_CUSTOM_MODEL:
    st.caption("Custom SpaCy NER model loaded.")
else:
    st.info(
        "Custom SpaCy model not found. Built-in phrase rules will detect common skills, "
        "degrees, and certifications; project and experience extraction requires a trained model."
    )

signature = _input_signature(jd_text, jd_skills, uploaded_files)
analyze_clicked = st.button(
    "Analyze resumes",
    type="primary",
    disabled=not jd_text.strip() or not uploaded_files,
)

if analyze_clicked:
    try:
        with st.spinner("Loading the embedding model and analyzing resumes..."):
            model = get_model()
            results, errors, target_skills = _analyze_resumes(
                jd_text, jd_skills, uploaded_files, model
            )
        st.session_state["analysis"] = {
            "signature": signature,
            "results": results,
            "errors": errors,
            "target_skills": target_skills,
        }
    except Exception as exc:
        st.error(f"Analysis could not start: {exc}")

analysis = st.session_state.get("analysis")
if analysis and analysis["signature"] != signature:
    st.info("Inputs changed. Analyze resumes again to refresh the results.")
elif analysis:
    for error in analysis["errors"]:
        st.warning(error)

    results = analysis["results"]
    if not results:
        st.warning("No resumes could be analyzed. Check the uploaded PDFs and try again.")
    else:
        if not analysis["target_skills"]:
            st.warning("No JD skills were provided or detected, so skill-gap results are empty.")
        elif not jd_skills:
            st.caption("Target skills were inferred from the job description.")

        dataframe = pd.DataFrame(results)
        st.subheader("Leaderboard")
        st.dataframe(
            dataframe[["Candidate", "Match Score (%)", "Matched Skills", "Missing Skills"]],
            use_container_width=True,
            hide_index=True,
        )

        average_score = dataframe["Match Score (%)"].mean()
        best_candidate = dataframe.iloc[0]["Filename"]
        metric_columns = st.columns(3)
        metric_columns[0].metric("Resumes analyzed", len(dataframe))
        metric_columns[1].metric("Average match", f"{average_score:.1f}%")
        metric_columns[2].metric("Top match", best_candidate)

        download_dataframe = dataframe.copy()
        list_columns = (
            "Extracted Skills", "Matched Skills", "Missing Skills", "Degrees",
            "Certifications", "Projects", "Experience",
        )
        for column in list_columns:
            download_dataframe[column] = download_dataframe[column].apply(", ".join)

        st.download_button(
            "Download results as CSV",
            download_dataframe.to_csv(index=False).encode("utf-8"),
            file_name="resume_match_results.csv",
            mime="text/csv",
        )

        st.subheader("Candidate details")
        selected_candidate = st.selectbox("Select a candidate", dataframe["Candidate"].tolist())
        candidate = next(row for row in results if row["Candidate"] == selected_candidate)
        left, right = st.columns(2)

        with left:
            st.metric("Semantic match score", f"{candidate['Match Score (%)']:.2f}%")
            st.write("**Degrees:**", ", ".join(candidate["Degrees"]) or "None detected")
            st.write("**Certifications:**", ", ".join(candidate["Certifications"]) or "None detected")
            st.write("**Projects:**", ", ".join(candidate["Projects"]) or "None detected")
            st.write("**Experience:**", ", ".join(candidate["Experience"]) or "None detected")

        with right:
            st.write("**Matched skills:**", ", ".join(candidate["Matched Skills"]) or "None")
            if candidate["Missing Skills"]:
                st.warning("**Missing JD skills:** " + ", ".join(candidate["Missing Skills"]))
            else:
                st.success("No listed JD skills are missing.")
            st.write("**All extracted skills:**", ", ".join(candidate["Extracted Skills"]) or "None detected")