import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Define folder paths
folder_path_jobs = "Job"  
folder_path_personnel = "Person"  
desired_levels_path = "شایستگی رفتاری روسای ادارات.xlsx"  
final_scores_path = "شایستگی رفتاری کارشناسان.xlsx"  

# Reading job files
all_jobs = []

# Iterate over job files in the specified folder
for file_name in os.listdir(folder_path_jobs):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(folder_path_jobs, file_name)
        df = pd.read_excel(file_path)
        # Check if necessary columns are present
        if "سمت" in df.columns and "شاخص" in df.columns:
            df = df[["سمت", "شاخص"]]  # Extract relevant columns
            all_jobs.append(df)

# Combine all job data into a single DataFrame
if all_jobs:
    jobs_df = pd.concat(all_jobs, ignore_index=True)
else:
    st.error("هیچ فایل شغلی معتبری یافت نشد.")
    st.stop()

# Reading desired levels
desired_levels_df = pd.read_excel(desired_levels_path)

# Ensure necessary columns exist
if "سمت" not in desired_levels_df.columns or "سطح مطلوب" not in desired_levels_df.columns:
    st.error("فایل سطح مطلوب شامل ستون‌های مناسب نیست.")
    st.stop()

# Group by job titles and calculate the mean of desired levels
desired_levels_grouped = desired_levels_df.groupby("سمت")["سطح مطلوب"].mean().reset_index()

# Reading personnel files
all_personnel_scores = []

# Iterate over personnel files in the specified folder
for file_name in os.listdir(folder_path_personnel):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(folder_path_personnel, file_name)
        df = pd.read_excel(file_path)
        # Check if necessary columns are present
        if "کد پرسنلی" in df.columns and "شاخص" in df.columns:
            df = df[["کد پرسنلی", "شاخص"]]
            all_personnel_scores.append(df)

# Combine all personnel data into a single DataFrame
if all_personnel_scores:
    personnel_scores_df = pd.concat(all_personnel_scores, ignore_index=True)
else:
    st.error("هیچ فایل پرسنلی معتبری یافت نشد.")
    st.stop()

# Reading final scores
final_scores_df = pd.read_excel(final_scores_path)

# Ensure necessary columns exist
if "کد پرسنلی" not in final_scores_df.columns or "نمره نهایی" not in final_scores_df.columns:
    st.error("فایل نمره نهایی شامل ستون‌های مناسب نیست.")
    st.stop()

# Group by personnel ID and calculate the mean of final scores
final_scores_df = final_scores_df.groupby("کد پرسنلی")["نمره نهایی"].mean().reset_index()

# Main UI layout
st.markdown(
    """
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    .stButton button {
        font-family: "B Nazanin";
        font-size: 18px;
    }
    .stTextInput label {
        font-family: "B Nazanin";
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# User input for job selection
st.markdown("#### نام شغل موردنظر خود را وارد کنید:")
selected_job = st.text_input("نام شغل:")

if selected_job:  
    # Check if the selected job exists in the job dataset
    if selected_job in jobs_df["سمت"].values:
        # Fetch the mean desired level for the selected job
        if selected_job in desired_levels_grouped["سمت"].values:
            mean_desired_level = desired_levels_grouped[desired_levels_grouped["سمت"] == selected_job]["سطح مطلوب"].values[0]
            st.info(f"📊 میانگین سطح مطلوب برای سمت '{selected_job}': {mean_desired_level:.2f}")
        else:
            st.warning("برای این سمت سطح مطلوبی تعریف نشده است.")

        # Fetch the criteria for the selected job
        selected_job_criteria = jobs_df[jobs_df["سمت"] == selected_job]["شاخص"].values
        if len(selected_job_criteria) > 0:
            vectorizer = TfidfVectorizer()
            job_vector = vectorizer.fit_transform(selected_job_criteria)

            # Calculate TF-IDF for personnel data
            personnel_vectors = vectorizer.transform(personnel_scores_df["شاخص"].astype(str))
            similarity_scores = cosine_similarity(job_vector, personnel_vectors)[0]

            # Add similarity scores to personnel data
            personnel_scores_df["امتیاز شباهت"] = similarity_scores

            # Merge with final scores to include final score
            merged_df = personnel_scores_df.merge(final_scores_df, on="کد پرسنلی", how="left")

            # Replace NaN values in 'نمره نهایی' with 0 for sorting
            merged_df["نمره نهایی"] = merged_df["نمره نهایی"].fillna(0)

            # Group and sort based on similarity and final scores
            top_personnel = (
                merged_df.groupby("کد پرسنلی")
                .agg({"امتیاز شباهت": "mean", "نمره نهایی": "mean"})
                .reset_index()
                .sort_values(by=["امتیاز شباهت", "نمره نهایی"], ascending=[False, False])  # Sorting
                .head(10)  # Select top 10
            )

            # Display top personnel with similarity score and final scores
            st.subheader("✅ افراد پیشنهادی برای سمت:")
            st.table(top_personnel)
        else:
            st.warning("شاخصی برای سمت انتخاب‌شده وجود ندارد.")
    else:
        st.error("سمت وارد شده در داده‌ها یافت نشد.")
