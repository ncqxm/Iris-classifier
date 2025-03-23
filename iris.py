import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# โหลดข้อมูล
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# เทรนโมเดล
@st.cache_resource
def load_models():
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    dt = DecisionTreeClassifier()
    dt.fit(X, y)

    return knn, dt

knn_model, dt_model = load_models()

now = datetime.fromtimestamp(time.time())
st.write("⏱️ ตอนนี้เวลา:", now.strftime("%Y-%m-%d %H:%M:%S"))


# Title
st.title("🌸 Iris Flower Classifier")
st.write("กรอกข้อมูลด้านล่างเพื่อทำนายสายพันธุ์ของดอกไม้ไอริส")

# Input จากผู้ใช้
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width  = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width  = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

# ทำนายผล
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
knn_pred = knn_model.predict(input_data)[0]
dt_pred = dt_model.predict(input_data)[0]

# โชว์ผลลัพธ์หลัก
st.success(f"KNN ทำนายว่า: **{target_names[knn_pred]}**")
st.info(f"Decision Tree ทำนายว่า: **{target_names[dt_pred]}**")

# ตารางเปรียบเทียบ
st.subheader("📊 เปรียบเทียบผลการทำนาย")
df_result = pd.DataFrame({
    "Model": ["K-Nearest Neighbors", "Decision Tree"],
    "Predicted Class": [target_names[knn_pred], target_names[dt_pred]]
})
st.table(df_result)

# Plot Scatter
st.subheader("📊 การกระจายตัวของข้อมูล (Scatter Plot)")
fig, ax = plt.subplots()

# วาดข้อมูลของชุดจริง
for i, target_name in enumerate(target_names):
    ax.scatter(X[y == i, 2], X[y == i, 3], label=target_name)

# วาดจุด input ของผู้ใช้
ax.scatter(petal_length, petal_width, color='red', s=100, edgecolors='black', label='Your Input')

# ตกแต่งกราฟ
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.legend()
st.pyplot(fig)

