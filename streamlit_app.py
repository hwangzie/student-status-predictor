# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from main import StudentStatusPredictor
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Student Status Predictor",
    page_icon="🎓",
    layout="wide"
)

# Initialize predictor
@st.cache_resource
def load_predictor():
    return StudentStatusPredictor()

def main():
    st.title("🎓 Student Status Prediction System")
    st.markdown("**Jaya Jaya Institut - Early Warning System**")
    
    try:
        predictor = load_predictor()
        
        # Sidebar for input
        st.sidebar.header("📝 Student Information")
        
        # Input fields
        age = st.sidebar.number_input("Age at Enrollment", min_value=16, max_value=60, value=20)
        admission_grade = st.sidebar.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=120.0)
        prev_grade = st.sidebar.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=120.0)
        
        # Semester 1
        st.sidebar.subheader("Semester 1")
        sem1_approved = st.sidebar.number_input("Units Approved (Sem 1)", min_value=0, max_value=10, value=5)
        sem1_grade = st.sidebar.number_input("Grade (Sem 1)", min_value=0.0, max_value=20.0, value=12.0)
        
        # Semester 2
        st.sidebar.subheader("Semester 2")
        sem2_approved = st.sidebar.number_input("Units Approved (Sem 2)", min_value=0, max_value=10, value=5)
        sem2_grade = st.sidebar.number_input("Grade (Sem 2)", min_value=0.0, max_value=20.0, value=12.0)
        
        # Financial status
        st.sidebar.subheader("Financial Status")
        tuition_updated = st.sidebar.selectbox("Tuition Fees Up to Date", ["Yes", "No"])
        is_debtor = st.sidebar.selectbox("Has Debt", ["No", "Yes"])
        scholarship = st.sidebar.selectbox("Scholarship Holder", ["No", "Yes"])
        
        # Economic indicators
        st.sidebar.subheader("Economic Indicators")
        unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 30.0, 10.0)
        inflation = st.sidebar.slider("Inflation Rate (%)", -5.0, 10.0, 2.0)
        
        # Prepare input data
        input_data = {
            'Age_at_enrollment': age,
            'Admission_grade': admission_grade,
            'Previous_qualification_grade': prev_grade,
            'Curricular_units_1st_sem_approved': sem1_approved,
            'Curricular_units_1st_sem_grade': sem1_grade,
            'Curricular_units_2nd_sem_approved': sem2_approved,
            'Curricular_units_2nd_sem_grade': sem2_grade,
            'Tuition_fees_up_to_date': 1 if tuition_updated == "Yes" else 0,
            'Debtor': 1 if is_debtor == "Yes" else 0,
            'Scholarship_holder': 1 if scholarship == "Yes" else 0,
            'Unemployment_rate': unemployment,
            'Inflation_rate': inflation
        }
        
        # Prediction button
        if st.sidebar.button("🔮 Predict Status", type="primary"):
            with st.spinner("Making prediction..."):
                result = predictor.predict(input_data)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Status", result['predicted_status'])
                    
                with col2:
                    st.metric("Confidence", result['confidence_percentage'])
                    
                with col3:
                    risk_color = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}
                    st.metric("Risk Level", f"{risk_color.get(result['risk_level'], '⚪')} {result['risk_level']}")
                
                # Recommendation
                st.info(f"**Recommendation:** {result['recommendation']}")
                
                # Probability chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(result['probabilities'].keys()),
                        y=list(result['probabilities'].values()),
                        marker_color=['red' if x == result['predicted_status'] else 'lightblue' 
                                    for x in result['probabilities'].keys()]
                    )
                ])
                fig.update_layout(title="Prediction Probabilities")
                st.plotly_chart(fig, use_container_width=True)
        
        # Sample data section
        st.markdown("---")
        st.subheader("📊 Sample Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚨 High Risk Student Example"):
                high_risk_data = {
                    'Age_at_enrollment': 22,
                    'Admission_grade': 110.0,
                    'Previous_qualification_grade': 100.0,
                    'Curricular_units_1st_sem_approved': 2,
                    'Curricular_units_1st_sem_grade': 8.5,
                    'Curricular_units_2nd_sem_approved': 1,
                    'Curricular_units_2nd_sem_grade': 9.0,
                    'Tuition_fees_up_to_date': 0,
                    'Debtor': 1,
                    'Scholarship_holder': 0,
                    'Unemployment_rate': 15.0,
                    'Inflation_rate': 3.0
                }
                result = predictor.predict(high_risk_data)
                st.write(f"**Status:** {result['predicted_status']}")
                st.write(f"**Confidence:** {result['confidence_percentage']}")
                st.write(f"**Risk:** {result['risk_level']}")
        
        with col2:
            if st.button("✅ Good Student Example"):
                good_data = {
                    'Age_at_enrollment': 18,
                    'Admission_grade': 150.0,
                    'Previous_qualification_grade': 160.0,
                    'Curricular_units_1st_sem_approved': 6,
                    'Curricular_units_1st_sem_grade': 15.5,
                    'Curricular_units_2nd_sem_approved': 6,
                    'Curricular_units_2nd_sem_grade': 16.0,
                    'Tuition_fees_up_to_date': 1,
                    'Debtor': 0,
                    'Scholarship_holder': 1,
                    'Unemployment_rate': 8.0,
                    'Inflation_rate': 1.0
                }
                result = predictor.predict(good_data)
                st.write(f"**Status:** {result['predicted_status']}")
                st.write(f"**Confidence:** {result['confidence_percentage']}")
                st.write(f"**Risk:** {result['risk_level']}")
                
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure all model files are available in the models/ directory")

if __name__ == "__main__":
    main()