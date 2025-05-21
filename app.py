import streamlit as st
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv("data/diabetes.csv")

# ui inputs
st.title("Diabetes Prediction")
models = ["Random Forest Classifier(Recommended)","Logistic Classification", "K-Nearest Neighbors Classification", "Decision Tree Classifier"]
st.sidebar.markdown("## Enter details")

glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=250, value=120)
pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20)
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=0)
bp = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=250, value=120)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, value=80)
skin_thickness = st.sidebar.slider("Skin Thickness", min_value=0, max_value=100, value=20)
bmi = st.sidebar.slider("BMI", min_value=0, max_value=70, value=30)
dpf = st.sidebar.slider("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
model_choice = st.selectbox("Choose a model", models)


model_info = {
    "Random Forest Classifier(Recommended)": "Accuracy: ~76.6%, ROC AUC: ~0.83",
    "Logistic Classification": "Accuracy: ~76.6%, ROC AUC: ~0.81",
    "K-Nearest Neighbors Classification": "Accuracy: ~69.5%, ROC AUC: ~0.77",
    "Decision Tree Classifier": "Accuracy: ~75.9%, ROC AUC: ~0.79"
}
st.markdown(f"**{model_info[model_choice]}**")


# graphs
with st.expander("Show Visualizations"):
    st.markdown("# Based on the data the models were trained on")
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("BMI vs Age (Colored by Diabetes Outcome)")
    fig2, ax2 = plt.subplots()
    colors = df['Outcome'].map({0: 'green', 1: 'red'})
    ax2.scatter(df['Age'], df['BMI'], c=colors, alpha=0.6)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("BMI")
    st.pyplot(fig2)


    st.subheader("Diabetes Pedigree Function Distribution")
    fig3, ax3 = plt.subplots()
    df[df['Outcome'] == 0]['DiabetesPedigreeFunction'].hist(alpha=0.5, bins=20, label='Non-Diabetic', ax=ax3)
    df[df['Outcome'] == 1]['DiabetesPedigreeFunction'].hist(alpha=0.5, bins=20, label='Diabetic', ax=ax3)
    ax3.set_xlabel("DPF")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("Diabetes Class Distribution")
    labels = ['Non-Diabetic', 'Diabetic']
    sizes = df['Outcome'].value_counts()
    fig4, ax4 = plt.subplots()
    ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff6666"])
    ax4.axis('equal')
    st.pyplot(fig4)



# loading models and scaler
scaler = joblib.load('scaler/scaler.pkl')

user_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
user_data_scaled = scaler.transform(user_data)

if model_choice == "Random Forest Classifier(Recommended)":
    model = joblib.load('models/random_forest_classifier.pkl')
elif model_choice == "Logistic Classification":
    model = joblib.load('models/logistic_regression.pkl')
elif model_choice == "K-Nearest Neighbors Classification":
    model = joblib.load('models/k_nearest_neighbors.pkl')
else:
    model = joblib.load('models/decision_tree_classifier.pkl')

# predict
if st.button("Predict"):
    prediction = model.predict(user_data_scaled)[0]
    if prediction == 1:
        st.error("The person **is likely diabetic.**")
    else:
        st.success("The person **might not be diabetic.**")