# ============================================================
# 🎓 SYLLABUS GAP ANALYZER (FINAL UPDATED - FIXED NLTK punkt_tab)
# ============================================================

import os
import re
import requests
import pandas as pd
import nltk
import plotly.express as px
from wordcloud import WordCloud
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import streamlit as st

# ============================================================
# 🧩 SETUP (UPDATED NLTK FIX)
# ============================================================

# Fix for new NLTK versions (needs punkt_tab)
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except Exception:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

st.set_page_config(page_title="Syllabus Gap Analyzer", layout="wide")

# ============================================================
# 📄 STEP 1 – Extract Syllabus Text from PDF
# ============================================================

def extract_syllabus_text(file):
    try:
        reader = PdfReader(file)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        text = " ".join(text_parts)
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

# ============================================================
# 🌐 STEP 2 – Get Job Descriptions (sample + fallback)
# ============================================================

def get_job_descriptions():
    urls = [
        "https://raw.githubusercontent.com/kaustubh-ai/datasets/main/ds_jobs_sample1.txt",
        "https://raw.githubusercontent.com/kaustubh-ai/datasets/main/ds_jobs_sample2.txt"
    ]
    text = ""
    for url in urls:
        try:
            r = requests.get(url, timeout=6)
            if r.status_code == 200 and r.text:
                text += r.text + " "
        except Exception:
            continue

    if len(text) < 800:
        text += (
            "We are looking for a Data Scientist with experience in Python, Machine Learning, "
            "Deep Learning, NLP, and cloud deployment (AWS, Azure, GCP). Skills in Tableau, "
            "Power BI, SQL, and Data Engineering are desirable."
        )

    return text

# ============================================================
# 🧹 STEP 3 – Clean and Preprocess Text
# ============================================================

def clean_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# ============================================================
# 📊 STEP 4 – Analyze Gaps using TF-IDF
# ============================================================

def analyze_gap(syllabus_text, job_text):
    docs = [syllabus_text, job_text]
    vectorizer = TfidfVectorizer(max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()
    df = pd.DataFrame(tfidf_matrix.toarray(), index=["Syllabus", "Jobs"], columns=features).T
    df["gap"] = df["Jobs"] - df["Syllabus"]
    df = df.sort_values(by="gap", ascending=False)
    return df

# ============================================================
# 🎨 STEP 5 – Visualization
# ============================================================

def visualize_interactive(df, syllabus_text_raw, job_text_raw, num_words=15, gap_threshold=0.02):
    st.subheader("📊 Interactive Visualizations")

    top_missing = df[df["gap"] > gap_threshold].head(num_words).reset_index().rename(columns={"index": "keyword"})

    # Bar chart
    if not top_missing.empty:
        fig_bar = px.bar(
            top_missing,
            x="gap",
            y="keyword",
            orientation="h",
            color="gap",
            color_continuous_scale="RdYlBu",
            title="Top Missing Topics in Syllabus",
            labels={"keyword": "Keyword", "gap": "TF-IDF Gap (Jobs – Syllabus)"},
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)

    # Wordclouds
    st.subheader("☁️ Word Clouds")
    c1, c2, c3 = st.columns(3)

    syllabus_wc = clean_text(syllabus_text_raw)
    job_wc = clean_text(job_text_raw)
    missing_wc = " ".join(top_missing["keyword"].astype(str).tolist())

    with c1:
        st.markdown("**Syllabus Word Cloud**")
        if syllabus_wc.strip():
            wc = WordCloud(width=600, height=400, background_color="white").generate(syllabus_wc)
            st.image(wc.to_array(), use_container_width=True)

    with c2:
        st.markdown("**Job Market Word Cloud**")
        if job_wc.strip():
            wc2 = WordCloud(width=600, height=400, background_color="white").generate(job_wc)
            st.image(wc2.to_array(), use_container_width=True)

    with c3:
        st.markdown("**Missing Topics Word Cloud**")
        if missing_wc.strip():
            wc3 = WordCloud(width=600, height=400, background_color="white").generate(missing_wc)
            st.image(wc3.to_array(), use_container_width=True)

# ============================================================
# 💾 STEP 6 – Export Reports
# ============================================================

def export_reports(df, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)

    df_export = df.reset_index().rename(columns={"index": "keyword"})

    all_path = os.path.join(out_dir, "all_keywords_report.csv")
    df_export.to_csv(all_path, index=False)

    missing = df_export[df_export["gap"] > 0.02]
    missing_path = os.path.join(out_dir, "missing_topics_report.csv")
    missing.to_csv(missing_path, index=False)

    excel_path = os.path.join(out_dir, "powerbi_data.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_export.to_excel(writer, sheet_name="All_Keywords", index=False)
        missing.to_excel(writer, sheet_name="Missing_Topics", index=False)

    summary_path = os.path.join(out_dir, "summary_report.pdf")
    c = canvas.Canvas(summary_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 760, "Syllabus Gap Analyzer - Summary Report")
    c.drawString(50, 740, f"Total Keywords: {len(df)}")
    c.drawString(50, 720, f"Missing Keywords (>0.02 gap): {len(missing)}")
    c.drawString(50, 700, "Top Missing Topics:")

    y = 680
    for kw in missing.head(10)["keyword"]:
        c.drawString(70, y, f"- {kw}")
        y -= 15
    c.save()

    return all_path, missing_path, excel_path, summary_path

# ============================================================
# 🚀 STREAMLIT APP UI
# ============================================================

st.title("🎓 Data Science Syllabus Gap Analyzer (Interactive + Learning Suggestions)")
st.write("Upload your syllabus PDF, compare trending job descriptions, and get improvement suggestions.")

uploaded_file = st.file_uploader("📄 Upload your syllabus PDF", type=["pdf"])

st.sidebar.header("⚙️ Controls")
num_words = st.sidebar.slider("Top missing topics to show", 5, 30, 12)
gap_threshold = st.sidebar.slider("Minimum TF-IDF gap threshold", 0.0, 0.2, 0.02, step=0.005)
show_raw = st.sidebar.checkbox("Show raw extracted syllabus text", value=False)

if uploaded_file:
    syllabus_text_raw = extract_syllabus_text(uploaded_file)
    job_text_raw = get_job_descriptions()

    if not syllabus_text_raw.strip():
        st.warning("⚠️ Could not extract text from this PDF. Try another file.")

    else:
        if show_raw:
            st.text_area("Extracted Syllabus Text", syllabus_text_raw[:6000], height=200)

        st.subheader("Job Description Preview")
        st.text_area("Job Data Sample", job_text_raw[:2000], height=120)

        if st.button("🔍 Run Gap Analysis"):
            syllabus_clean = clean_text(syllabus_text_raw)
            job_clean = clean_text(job_text_raw)
            df = analyze_gap(syllabus_clean, job_clean)

            all_csv, missing_csv, excel_path, pdf_path = export_reports(df)

            st.success("✅ Analysis Complete!")
            st.dataframe(df.head(20))

            st.subheader("📁 Download Reports")
            with open(all_csv, "rb") as f:
                st.download_button("All Keywords CSV", f, file_name="all_keywords_report.csv")
            with open(missing_csv, "rb") as f:
                st.download_button("Missing Topics CSV", f, file_name="missing_topics_report.csv")
            with open(excel_path, "rb") as f:
                st.download_button("Power BI Excel Data", f, file_name="powerbi_data.xlsx")
            with open(pdf_path, "rb") as f:
                st.download_button("Summary PDF", f, file_name="summary_report.pdf")

            visualize_interactive(df, syllabus_text_raw, job_text_raw, num_words=num_words, gap_threshold=gap_threshold)

else:
    st.info("📤 Please upload a syllabus PDF to begin.")

# ============================================================
# 🎯 END OF FILE
# ============================================================
