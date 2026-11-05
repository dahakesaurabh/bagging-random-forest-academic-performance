import pickle
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Random Forest Classifier App",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("RF_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("🎓 Student Performance Prediction App")
st.write("Enter the student details below to get the prediction from the Random Forest model.")

with st.form("prediction_form"):
    st.subheader("Input Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=10, max_value=100, value=18, step=1)
        study_hours_per_week = st.number_input("Study Hours Per Week", min_value=0.0, max_value=168.0, value=10.0, step=0.5)
        attendance_rate = st.number_input("Attendance Rate (0-1 or %)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        parent_education = st.selectbox("Parent Education", ["High School", "Bachelor", "Master", "PhD", "Other"])
        
    with col2:
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        final_score = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
   
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'study_hours_per_week': [study_hours_per_week],
        'attendance_rate': [attendance_rate],
        'parent_education': [parent_education],
        'internet_access': [internet_access],
        'extracurricular': [extracurricular],
        'previous_score': [previous_score],
        'final_score': [final_score]
    })
    
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
        
        st.success(f"### Prediction Result: **{prediction[0]}**")
        
        if prediction_proba is not None:
            st.write(f"**Confidence / Probabilities:**")
            classes = model.classes_
            for cls, prob in zip(classes, prediction_proba[0]):
                st.write(f"- Class `{cls}`: `{prob:.2f}`")
                
    except Exception as e:
      st.error(f"Error during prediction: {e}")
      st.info("Note: If your model requires categorical encoding (like OneHotEncoding or LabelEncoding) that was baked into a pipeline rather than raw strings, make sure your pipeline object is what was saved inside `RF_model.pkl`.")
