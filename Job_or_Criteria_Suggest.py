import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import streamlit as st

# 1. Load Excel files
folder_path = "excel_files"  # Path to the folder containing Excel files
all_data = []

# Read Excel files
for file_name in os.listdir(folder_path):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path)
        # Ensure required columns are present
        if "سمت" in df.columns and "شاخص" in df.columns:
            df = df[["سمت", "شاخص"]]
            all_data.append(df)
        else:
            print(f"Skipped {file_name}: Columns not found")

# Combine all data
if all_data:
    df = pd.concat(all_data, ignore_index=True)
else:
    st.error("No valid files with required columns found")
    st.stop()

# 2. Text preprocessing and vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["شاخص"])

# 3. Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Set up the UI
st.set_page_config(page_title="Job Suggestion System", layout="wide")
st.title("🎯 Job or Criteria Suggestion System")
st.markdown(".This tool helps you find suitable jobs or related criteria based on your input")

# Right-to-left alignment styling
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

# Choose the search type
search_type = st.radio("Choose the type of search:", ["Search by Criteria", "Search by Job Title"])

if search_type == "Search by Criteria":
    st.markdown("#### Enter your criteria, separated by commas (,):")
    user_input = st.text_input("Criteria:", "")
    if user_input:
        # Convert input into a list
        input_list = [item.strip() for item in user_input.split(",")]

        # Compute similarities
        similarities = []
        for user_text in input_list:
            user_vector = vectorizer.transform([user_text])
            similarity = cosine_similarity(user_vector, X)
            similarities.append(similarity[0])

        # Calculate average similarity
        average_similarity = sum(similarities) / len(input_list)
        df['Average_Similarity'] = average_similarity

        # Sort and display suggested jobs
        suggested_jobs = df.sort_values(by="Average_Similarity", ascending=False)
        st.markdown("### ✅ Suggested Jobs:")
        for index, row in suggested_jobs.drop_duplicates(subset=["سمت"]).head(10).iterrows():
            st.success(f"**{row['سمت']}** (Similarity Score: {row['Average_Similarity']:.2f})")

elif search_type == "Search by Job Title":
    st.markdown("#### Enter the job title:")
    job_input = st.text_input("Job Title:", "")
    if job_input:
        # Filter data for the given job title
        filtered_data = df[df["سمت"].str.contains(job_input, case=False, na=False)]

        if not filtered_data.empty:
            st.markdown("### ✅ Related Criteria:")
            for index, row in filtered_data.iterrows():
                st.info(f"**{row['شاخص']}**")
        else:
            st.warning("Job not found. Please enter a more specific title.")
