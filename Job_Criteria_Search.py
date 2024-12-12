import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import streamlit as st

# 1. Load Excel files
folder_path_jobs = "excel_files"  
folder_path_personnel = "personal_id/2.xlsx" 
final_scores_file = "personal_id/6.xlsx" 

# Load job data
all_jobs = []
for file_name in os.listdir(folder_path_jobs):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(folder_path_jobs, file_name)
        df = pd.read_excel(file_path)
        if "سمت" in df.columns and "شاخص" in df.columns:
            df = df[["سمت", "شاخص"]]
            all_jobs.append(df)

# Combine all job data
if all_jobs:
    jobs_df = pd.concat(all_jobs, ignore_index=True)
else:
    st.error("هیچ فایل شغلی معتبری پیدا نشد.")
    st.stop()

# Load personnel data
all_personnel = []
df = pd.read_excel(folder_path_personnel)
if "کد پرسنلی" in df.columns and "شاخص" in df.columns:
    df = df[["کد پرسنلی", "شاخص"]]
    all_personnel.append(df)

# Combine all personnel data
if all_personnel:
    personnel_df = pd.concat(all_personnel, ignore_index=True)
else:
    st.error("error while uploading files")
    st.stop()

# Remove duplicate personnel codes and criteria
personnel_df = personnel_df.drop_duplicates(subset=["کد پرسنلی", "شاخص"])

# Load final scores data
if os.path.exists(final_scores_file):
    final_scores_df = pd.read_excel(final_scores_file)
    if {"کد پرسنلی", "نمره نهایی"}.issubset(final_scores_df.columns):
        # Calculate the mean score for each personnel code
        final_scores_mean = final_scores_df.groupby("کد پرسنلی")["نمره نهایی"].mean().reset_index()
        final_scores_mean = final_scores_mean.rename(columns={"نمره": "نمره نهایی"})
    else:
        st.error("error while uploading files")
        st.stop()
else:
    st.error("error while uploading files")
    st.stop()

# 2. Vectorize criteria
vectorizer = TfidfVectorizer()
jobs_X = vectorizer.fit_transform(jobs_df["شاخص"])
personnel_X = vectorizer.transform(personnel_df["شاخص"])

# Set up the UI
st.set_page_config(page_title="سیستم پیشنهاد شغل", layout="wide")
st.title("🎯 سیستم پیشنهاد فرد مناسب")
st.markdown(".این ابزار به شما کمک می‌کند افراد مناسب برای شاخص‌های موردنظر را پیدا کنید.")

# CSS for right-to-left alignment
st.markdown(
    """
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Get user input
st.markdown("#### شاخص‌های خود را با کاما (,) از هم جدا کنید:")
user_input = st.text_input("شاخص‌ها:", "")
if user_input:
    # Convert input into a list
    input_list = [item.strip() for item in user_input.split(",")]
    user_vector = vectorizer.transform([", ".join(input_list)])

    # Compute similarities
    similarity_scores = cosine_similarity(user_vector, personnel_X)[0]

    # Add similarity scores to personnel data
    personnel_df["امتیاز شباهت"] = similarity_scores

    # Remove duplicate personnel codes
    personnel_df = personnel_df.drop_duplicates(subset=["کد پرسنلی"])

    # Merge with final scores data
    result_df = pd.merge(personnel_df, final_scores_mean, on="کد پرسنلی", how="left")

    # Sort by similarity score and final score
    result_df = result_df.sort_values(by=["امتیاز شباهت", "نمره نهایی"], ascending=[False, False]).head(10)

    # Display results
    st.markdown("### ✅ افراد پیشنهادی:")
    for index, row in result_df.iterrows():
        st.success(
            f"**کد پرسنلی: {row['کد پرسنلی']}** "
            f"(امتیاز شباهت: {row['امتیاز شباهت']:.2f}, نمره نهایی: {row['نمره نهایی']:.2f})"
        )
