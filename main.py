import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
import warnings
import os
warnings.filterwarnings('ignore')

class StudentStatusPredictor:
    """
    Sistem prediksi status mahasiswa menggunakan machine learning
    """
    
    def __init__(self, model_path: str = './models'):
        """
        Initialize predictor dengan model dan preprocessing objects yang telah disimpan
        
        Args:
            model_path (str): Path ke direktori yang berisi model yang disimpan
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.model_info = None
        self.load_models()
    
    def load_models(self):
        """Load model dan preprocessing objects yang telah disimpan"""
        try:
            print("🔄 Loading model artifacts...")
            
            # Load model components
            self.model = joblib.load(f'{self.model_path}/best_model.pkl')
            self.scaler = joblib.load(f'{self.model_path}/scaler.pkl')
            self.label_encoders = joblib.load(f'{self.model_path}/label_encoders.pkl')
            self.feature_names = joblib.load(f'{self.model_path}/feature_names.pkl')
            self.model_info = joblib.load(f'{self.model_path}/model_info.pkl')
            
            print("✅ Models loaded successfully!")
            print(f"📊 Model Type: {self.model_info['model_type']}")
            print(f"📊 Model Name: {self.model_info['model_name']}")
            print(f"📊 Accuracy: {self.model_info['accuracy']:.4f}")
            print(f"📊 Features: {self.model_info['n_features']}")
            print(f"📊 Classes: {self.model_info['target_classes']}")
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")
            print("Please run the training notebook first to generate model files.")
            raise
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def get_feature_template(self) -> Dict:
        """
        Mengembalikan template dengan semua fitur yang diperlukan
        
        Returns:
            Dict: Template fitur dengan nilai default
        """
        template = {}
        for feature in self.feature_names:
            template[feature] = 0  # Default value
        
        return template
    
    def preprocess_input(self, input_data: Dict) -> np.ndarray:
        """
        Preprocess input data agar sesuai dengan format training
        
        Args:
            input_data (Dict): Dictionary yang berisi nilai fitur
            
        Returns:
            np.ndarray: Data yang telah dipreprocess untuk prediksi
        """
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Pastikan semua fitur yang diperlukan ada
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value untuk fitur yang hilang
        
        # Reorder kolom sesuai dengan training data
        df = df[self.feature_names]
        
        # Apply label encoding untuk categorical features
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle kategori yang tidak dikenal
                    print(f"⚠️ Unknown category in {col}, using default value")
                    df[col] = 0
        
        # Handle missing values
        df = df.fillna(0)
        
        # Scale features jika model memerlukan scaling
        if self.model_info['model_name'] in ['SVM', 'Logistic Regression']:
            return self.scaler.transform(df)
        else:  # Random Forest
            return df.values
    
    def predict(self, input_data: Dict) -> Dict:
        """
        Membuat prediksi pada input data
        
        Args:
            input_data (Dict): Dictionary yang berisi nilai fitur
            
        Returns:
            Dict: Hasil prediksi dengan skor probabilitas
        """
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Get confidence score
            confidence = float(max(probabilities))
            
            # Create probability dictionary
            prob_dict = {
                class_name: float(prob) 
                for class_name, prob in zip(self.model.classes_, probabilities)
            }
            
            # Determine risk level
            risk_level = self._get_risk_level(prediction, confidence)
            
            result = {
                'predicted_status': prediction,
                'confidence': confidence,
                'confidence_percentage': f"{confidence:.1%}",
                'risk_level': risk_level,
                'probabilities': prob_dict,
                'recommendation': self._get_recommendation(prediction, confidence)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            raise
    
    def _get_risk_level(self, prediction: str, confidence: float) -> str:
        """Menentukan tingkat risiko berdasarkan prediksi dan confidence"""
        if prediction == 'Dropout':
            if confidence > 0.8:
                return 'High Risk'
            elif confidence > 0.6:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        else:
            return 'Low Risk'
    
    def _get_recommendation(self, prediction: str, confidence: float) -> str:
        """Memberikan rekomendasi berdasarkan hasil prediksi"""
        if prediction == 'Dropout':
            if confidence > 0.8:
                return "🚨 Urgent: Student needs immediate intervention and support"
            elif confidence > 0.6:
                return "⚠️ Warning: Monitor student progress closely and provide additional support"
            else:
                return "💡 Info: Consider preventive measures and regular check-ins"
        elif prediction == 'Enrolled':
            return "📚 Student is progressing. Continue monitoring and provide support as needed"
        else:  # Graduate
            return "🎓 Student is on track for graduation. Maintain current support level"
    
    def predict_batch(self, input_list: List[Dict]) -> List[Dict]:
        """
        Membuat prediksi pada multiple inputs
        
        Args:
            input_list (List[Dict]): List dari input dictionaries
            
        Returns:
            List[Dict]: List dari hasil prediksi
        """
        results = []
        for i, input_data in enumerate(input_list):
            try:
                result = self.predict(input_data)
                result['student_id'] = i + 1
                results.append(result)
            except Exception as e:
                print(f"❌ Error predicting student {i+1}: {e}")
                results.append({
                    'student_id': i + 1,
                    'error': str(e),
                    'predicted_status': 'Error',
                    'confidence': 0.0
                })
        return results
    
    def analyze_features(self, input_data: Dict) -> Dict:
        """
        Analisis pentingnya fitur untuk prediksi (hanya untuk Random Forest)
        
        Args:
            input_data (Dict): Input data
            
        Returns:
            Dict: Analisis fitur
        """
        if self.model_info['model_name'] != 'Random Forest':
            return {"message": "Feature analysis only available for Random Forest models"}
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        # Create feature importance dictionary
        importance_dict = {
            feature: float(importance) 
            for feature, importance in zip(self.feature_names, feature_importance)
        }
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_10_features': dict(sorted_features[:10]),
            'all_features': importance_dict
        }

def main():
    """Main function untuk demonstrasi predictor"""
    print("🎓 Student Status Prediction System")
    print("Developed by: Muhammad Za'im Shidqi")
    print("="*60)
    
    try:
        # Initialize predictor
        predictor = StudentStatusPredictor()
        
        # Example input data - mahasiswa dengan profil berisiko
        high_risk_student = {
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
        
        # Example input data - mahasiswa dengan profil baik
        good_student = {
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
        
        # Make predictions
        print("\n📊 Sample Predictions:")
        print("-" * 40)
        
        print("\n1. High Risk Student Profile:")
        result1 = predictor.predict(high_risk_student)
        print(f"   Status: {result1['predicted_status']}")
        print(f"   Confidence: {result1['confidence_percentage']}")
        print(f"   Risk Level: {result1['risk_level']}")
        print(f"   Recommendation: {result1['recommendation']}")
        
        print("\n2. Good Student Profile:")
        result2 = predictor.predict(good_student)
        print(f"   Status: {result2['predicted_status']}")
        print(f"   Confidence: {result2['confidence_percentage']}")
        print(f"   Risk Level: {result2['risk_level']}")
        print(f"   Recommendation: {result2['recommendation']}")
        
        # Batch prediction example
        print("\n📈 Batch Prediction Example:")
        batch_results = predictor.predict_batch([high_risk_student, good_student])
        for result in batch_results:
            print(f"   Student {result['student_id']}: {result['predicted_status']} ({result.get('confidence_percentage', 'N/A')})")
        
        # Feature analysis
        print("\n🔍 Feature Analysis (Top 5):")
        feature_analysis = predictor.analyze_features(good_student)
        if 'top_10_features' in feature_analysis:
            for i, (feature, importance) in enumerate(list(feature_analysis['top_10_features'].items())[:5]):
                print(f"   {i+1}. {feature}: {importance:.4f}")
        
        print("\n" + "="*60)
        print("✅ Prediction system working successfully!")
        print("💡 Ready for integration into larger systems")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure you have run the training notebook and saved the models first.")

if __name__ == "__main__":
    main()