import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import streamlit as st

# 1. بارگذاری فایل‌های اکسل
folder_path = "excel_files"  # مسیر پوشه فایل‌های اکسل
all_data = []

# خواندن فایل‌های اکسل
for file_name in os.listdir(folder_path):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path)
        if "سمت" in df.columns and "شاخص" in df.columns:  # بررسی وجود ستون‌ها
            df = df[["سمت", "شاخص"]]
            all_data.append(df)
        else:
            print(f"Skipped {file_name}: Columns not found")

# ترکیب تمام داده‌ها
if all_data:
    df = pd.concat(all_data, ignore_index=True)
else:
    st.error("هیچ فایل معتبری با ستون‌های موردنیاز پیدا نشد")
    st.stop()

# 2. پیش‌پردازش متن و بردارسازی
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["شاخص"])

# 3. خوشه‌بندی
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# طراحی رابط کاربری
st.set_page_config(page_title="سیستم پیشنهاد شغل", layout="wide")
st.title("🎯 سیستم پیشنهاد شغل یا شاخص")
st.markdown(".این ابزار به شما کمک می‌کند بر اساس ورودی، شغل مناسب یا شاخص‌های مرتبط را پیدا کنید")

# انتخاب نوع جستجو
search_type = st.radio("نوع جستجو را انتخاب کنید:", ["جستجو بر اساس شاخص", "جستجو بر اساس شغل"])

if search_type == "جستجو بر اساس شاخص":
    st.markdown("#### شاخص‌های خود را با کاما (,) از هم جدا کنید:")
    user_input = st.text_input("شاخص‌ها:", "")
    if user_input:
        # تبدیل ورودی‌ها به لیست
        input_list = [item.strip() for item in user_input.split(",")]

        # محاسبه شباهت‌ها
        similarities = []
        for user_text in input_list:
            user_vector = vectorizer.transform([user_text])
            similarity = cosine_similarity(user_vector, X)
            similarities.append(similarity[0])

        # محاسبه میانگین شباهت
        average_similarity = sum(similarities) / len(input_list)
        df['Average_Similarity'] = average_similarity

        # مرتب‌سازی و نمایش شغل‌های پیشنهادی
        suggested_jobs = df.sort_values(by="Average_Similarity", ascending=False)
        st.markdown("### ✅ شغل‌های پیشنهادی:")
        for index, row in suggested_jobs.drop_duplicates(subset=["سمت"]).head(10).iterrows():
            st.success(f"**{row['سمت']}** (امتیاز شباهت: {row['Average_Similarity']:.2f})")

elif search_type == "جستجو بر اساس شغل":
    st.markdown("#### عنوان شغل خود را وارد کنید:")
    job_input = st.text_input("شغل:", "")
    if job_input:
        # فیلتر داده‌ها برای شغل موردنظر
        filtered_data = df[df["سمت"].str.contains(job_input, case=False, na=False)]

        if not filtered_data.empty:
            st.markdown("### ✅ شاخص‌های مرتبط:")
            for index, row in filtered_data.iterrows():
                st.info(f"**{row['شاخص']}**")
        else:
            st.warning(".شغل موردنظر پیدا نشد. لطفاً عنوان دقیق‌تری وارد کنید")
