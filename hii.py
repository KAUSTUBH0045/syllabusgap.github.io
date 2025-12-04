
# ============================================================
# 🎓 SYLLABUS GAP ANALYZER (FINAL UPDATED)
# Interactive + Power BI export + Learning Suggestions (images + durations)
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
# 🧩 SETUP
# ============================================================
# Download NLTK resources if missing (quiet)
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

st.set_page_config(page_title="Syllabus Gap Analyzer", layout="wide")

# ============================================================
# 📄 STEP 1 – Extract Syllabus Text from PDF (safer)
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
# 🌐 STEP 2 – Get Job Descriptions (sample remote + fallback)
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
            # ignore network issues, continue
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
# 🎨 STEP 5 – Visualization (Interactive + WordClouds + fixes)
# ============================================================
def visualize_interactive(df, syllabus_text_raw, job_text_raw, num_words=15, gap_threshold=0.02):
    st.subheader("📊 Interactive Visualizations")

    # Top missing topics table
    top_missing = df[df["gap"] > gap_threshold].head(num_words).reset_index().rename(columns={"index": "keyword"})

    # --- Bar Chart (Top Missing Topics)
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
    else:
        st.info("No topics exceed the chosen gap threshold — increase sensitivity or lower threshold.")

    # --- Word Clouds (Syllabus, Jobs, Missing)
    st.subheader("☁️ Word Clouds")
    c1, c2, c3 = st.columns(3)
    # Syllabus wordcloud from raw text (cleaned)
    syllabus_wc_text = clean_text(syllabus_text_raw)
    jobs_wc_text = clean_text(job_text_raw)
    missing_wc_text = " ".join(top_missing["keyword"].astype(str).tolist()) if not top_missing.empty else ""

    with c1:
        st.markdown("**Syllabus Word Cloud**")
        if syllabus_wc_text.strip():
            wc = WordCloud(width=600, height=400, background_color="white").generate(syllabus_wc_text)
            st.image(wc.to_array(), use_container_width=True)
        else:
            st.info("No syllabus text to generate wordcloud.")

    with c2:
        st.markdown("**Job Market Word Cloud**")
        if jobs_wc_text.strip():
            wc2 = WordCloud(width=600, height=400, background_color="white").generate(jobs_wc_text)
            st.image(wc2.to_array(), use_container_width=True)
        else:
            st.info("No job text to generate wordcloud.")

    with c3:
        st.markdown("**Missing Topics Word Cloud**")
        if missing_wc_text.strip():
            wc3 = WordCloud(width=600, height=400, background_color="white").generate(missing_wc_text)
            st.image(wc3.to_array(), use_container_width=True)
        else:
            st.info("No missing topics to show here (increase sensitivity).")

    # --- Heatmap (top keywords)
    st.subheader("🔁 TF-IDF Heatmap (Top 20 keywords)")
    try:
        heat_df = df.head(20)[["Syllabus", "Jobs"]]
        fig_heat = px.imshow(heat_df.T, text_auto=True, labels=dict(x="Keyword", y="Source", color="TF-IDF"))
        fig_heat.update_xaxes(tickangle=45)
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception:
        st.info("Heatmap could not be rendered for the current data.")

    # --- Bubble Chart (correlation) — ensure positive sizes
    st.subheader("🔵 Keyword Importance Bubble Chart")
    df_display = df.head(50).copy().reset_index().rename(columns={"index": "keyword"})
    # size must be positive -> use absolute gap, scaled, and ensure min size > 0
    df_display["size_val"] = (df_display["gap"].abs() * 100).clip(lower=1)
    fig_bubble = px.scatter(
        df_display,
        x="Syllabus",
        y="Jobs",
        size="size_val",
        color="gap",
        hover_name="keyword",
        color_continuous_scale="Viridis",
        title="Syllabus TF-IDF vs Job TF-IDF — bubble size ∝ |gap|",
        labels={"Syllabus": "Syllabus TF-IDF", "Jobs": "Job TF-IDF", "gap": "Jobs - Syllabus (gap)"},
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

# ============================================================
# 💾 STEP 6 – Export Reports (CSV, PDF, Excel for Power BI)
# ============================================================
def export_reports(df, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)

    df_export = df.reset_index().rename(columns={"index": "keyword"})
    all_path = os.path.join(out_dir, "all_keywords_report.csv")
    df_export.to_csv(all_path, index=False)

    missing = df_export[df_export["gap"] > 0.02]
    missing_path = os.path.join(out_dir, "missing_topics_report.csv")
    missing.to_csv(missing_path, index=False)

    # Excel export (Power BI ready)
    excel_path = os.path.join(out_dir, "powerbi_data.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_export.to_excel(writer, sheet_name="All_Keywords", index=False)
        missing.to_excel(writer, sheet_name="Missing_Topics", index=False)

    # PDF summary
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
# 💡 STEP 7 – Personalized Learning Recommendations (images + estimations)
# ============================================================
def recommend_learning(df):
    st.subheader("🎯 Personalized Learning Suggestions")
    missing = df[df["gap"] > 0.02].head(12)  # show top 12 suggestions

    if missing.empty:
        st.success("🎉 Your syllabus already covers most trending topics — great job!")
        return

    st.info("Click a skill to expand a short learning plan, an estimated time, and quick resources.")

    # helper: estimate time
    def estimate_time(keyword):
        k = keyword.lower()
        if any(x in k for x in ["python", "r", "sql"]):
            return "⏱️ ~3 weeks (programming fundamentals + practice projects)"
        if any(x in k for x in ["machine learning", "deep learning", "neural", "ai", "nlp", "transformer"]):
            return "⏱️ ~6–8 weeks (ML/DL fundamentals + practical projects)"
        if any(x in k for x in ["tableau", "powerbi", "visualization", "plotly"]):
            return "⏱️ ~2–3 weeks (dashboards + storytelling)"
        if any(x in k for x in ["aws", "azure", "gcp", "cloud", "spark", "hadoop"]):
            return "⏱️ ~5–6 weeks (cloud basics + hands-on labs)"
        if any(x in k for x in ["git", "excel", "api"]):
            return "⏱️ ~1–2 weeks (tooling and practice)"
        return "⏱️ ~2–4 weeks (general upskilling + small projects)"

    # simple logo map (public logos from wiki or common svg/png)
    image_map = {
        "python": "https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg",
        "machine": "https://upload.wikimedia.org/wikipedia/commons/1/14/Deep_learning_logo.png",
        "nlp": "https://upload.wikimedia.org/wikipedia/commons/9/9a/NLP_logo.png",
        "powerbi": "https://upload.wikimedia.org/wikipedia/commons/c/cf/New_Power_BI_Logo.svg",
        "tableau": "https://upload.wikimedia.org/wikipedia/commons/4/4b/Tableau_Logo.png",
        "aws": "https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg",
        "azure": "https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg",
        "sql": "https://upload.wikimedia.org/wikipedia/commons/8/87/Sql_data_base_with_logo.png",
        "git": "https://upload.wikimedia.org/wikipedia/commons/3/3f/Git_icon.svg",
        "excel": "https://upload.wikimedia.org/wikipedia/commons/7/73/Microsoft_Excel_2013_logo.svg",
    }

    for keyword in missing.index:
        display_name = keyword.title()
        est = estimate_time(keyword)
        # choose image by matching keyword tokens to image_map keys
        image_url = None
        for key, url in image_map.items():
            if key in keyword.lower():
                image_url = url
                break

        with st.expander(f"🔹 {display_name} — {est}"):
            cols = st.columns([1, 4])
            with cols[0]:
                if image_url:
                    st.image(image_url, width=80)
                else:
                    # fallback small placeholder
                    st.write("🧩")
            with cols[1]:
                st.markdown(f"**Why learn {display_name}?**\n\nThis skill shows up strongly in job descriptions but is underrepresented in your syllabus.")
                st.markdown(
                    f"**Estimated learning time:** {est}\n\n"
                    "- **Suggested learning path:** Quick tutorials → Hands-on mini-project → Build/host portfolio demo\n"
                    "- **Resources:** [Coursera](https://www.coursera.org), [Kaggle Learn](https://www.kaggle.com/learn), [Udemy](https://www.udemy.com), [YouTube tutorials](https://www.youtube.com/results?search_query={keyword}+tutorial)\n"
                )
                st.markdown("---")
                # small action buttons (open links) — Streamlit doesn't support direct external opening, but we provide links
                st.write("Quick links:")
                st.markdown(f"- [Search courses for {display_name} on Coursera](https://www.coursera.org/search?query={keyword})")
                st.markdown(f"- [Search {display_name} on Kaggle Learn](https://www.kaggle.com/learn)")
                st.markdown(f"- [YouTube: {display_name} tutorials](https://www.youtube.com/results?search_query={keyword}+tutorial)")

    st.markdown(
        """
        ---
        💬 **Note:** These time estimates are approximate for a focused learner (self-study + hands-on practice).
        Real time will vary depending on background and depth of learning desired.
        """
    )

# ============================================================
# 🚀 STEP 8 – STREAMLIT FRONT-END (main)
# ============================================================
st.title("🎓 Data Science Syllabus Gap Analyzer (Interactive + Learning Suggestions)")
st.write("Upload your syllabus PDF, compare against trending job descriptions, and get personalized learning suggestions with estimated times and resources.")

uploaded_file = st.file_uploader("📄 Upload your syllabus PDF", type=["pdf"])

# sidebar controls
st.sidebar.header("⚙️ Controls")
num_words = st.sidebar.slider("Top missing topics to show", 5, 30, 12)
gap_threshold = st.sidebar.slider("Minimum TF-IDF gap threshold", 0.0, 0.2, 0.02, step=0.005)
show_raw = st.sidebar.checkbox("Show raw extracted syllabus text", value=False)

if uploaded_file:
    syllabus_text_raw = extract_syllabus_text(uploaded_file)
    job_text_raw = get_job_descriptions()

    if not syllabus_text_raw.strip():
        st.warning("⚠️ Could not extract text from the uploaded PDF. Try another file or ensure it's text-based (not scanned).")
    else:
        st.subheader("1️⃣ Syllabus Preview (extracted)")
        if show_raw:
            st.text_area("Extracted Syllabus Text", syllabus_text_raw[:6000], height=200)
        else:
            st.write("Syllabus extracted — toggle **Show raw extracted syllabus text** in the sidebar to inspect.")

        st.subheader("2️⃣ Job Descriptions Sample")
        st.text_area("Sample Job Data (preview)", job_text_raw[:2000], height=120)

        if st.button("🔍 Run Gap Analysis"):
            with st.spinner("Processing — cleaning text and computing TF-IDF..."):
                syllabus_clean = clean_text(syllabus_text_raw)
                job_clean = clean_text(job_text_raw)

                df = analyze_gap(syllabus_clean, job_clean)

                # Export reports
                all_csv, missing_csv, excel_path, pdf_path = export_reports(df)

            st.success("✅ Analysis complete!")
            st.subheader("Top Keywords (sample)")
            st.dataframe(df.head(20))

            st.subheader("📁 Download Reports")
            with open(all_csv, "rb") as f:
                st.download_button("Download All Keywords CSV", f, file_name="all_keywords_report.csv")
            with open(missing_csv, "rb") as f:
                st.download_button("Download Missing Topics CSV", f, file_name="missing_topics_report.csv")
            with open(excel_path, "rb") as f:
                st.download_button("Download Power BI Excel Data", f, file_name="powerbi_data.xlsx")
            with open(pdf_path, "rb") as f:
                st.download_button("Download Summary PDF", f, file_name="summary_report.pdf")

            # Visualize
            visualize_interactive(df, syllabus_text_raw, job_text_raw, num_words=num_words, gap_threshold=gap_threshold)
            recommend_learning(df)

else:
    st.info("📤 Please upload a syllabus PDF to begin analysis.")

# ============================================================
# 🎯 END OF FILE
# ============================================================
