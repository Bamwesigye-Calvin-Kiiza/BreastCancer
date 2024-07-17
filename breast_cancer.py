import os
import joblib
import pandas as pd
import streamlit as st

# Load the label encoders and scalers
def load_objects():
    global label_encoders, label_encoders_3, scalers, scaler, log_reg_model
    
    label_encoders = {}
    label_encoders_path = 'label_encoders'
    for file_name in os.listdir(label_encoders_path):
        if file_name.endswith('.pkl'):
            path = os.path.join(label_encoders_path, file_name)
            column_name = file_name.replace('.pkl', '')
            label_encoders[column_name] = joblib.load(path)

    label_encoders_3 = {}
    label_encoders_3_path = 'label_encoders_3'
    for file_name in os.listdir(label_encoders_3_path):
        if file_name.endswith('.pkl'):
            path = os.path.join(label_encoders_3_path, file_name)
            column_name = file_name.replace('.pkl', '')
            label_encoders_3[column_name] = joblib.load(path)

    scalers = {}
    scalers_path = 'scalers'
    for file_name in os.listdir(scalers_path):
        if file_name.endswith('.pkl'):
            path = os.path.join(scalers_path, file_name)
            column_name = file_name.replace('.pkl', '')
            scalers[column_name] = joblib.load(path)

    scaler_path = 'scalers/sc.pkl'
    scaler = joblib.load(scaler_path)
    
    model_path = 'logistic_regression_model_2.pkl'
    log_reg_model = joblib.load(model_path)

# Define the function to make predictions
def predict(input_data):
    # Convert the dictionary to a DataFrame
    new_data_df = pd.DataFrame([input_data])
    
    # Apply the saved label encoders to the new data
    for column, encoder in label_encoders.items():
        if column in new_data_df.columns:
            try:
                new_labels = new_data_df[column].unique()
                known_labels = encoder.classes_
                unknown_labels = [label for label in new_labels if label not in known_labels]
                if unknown_labels:
                    ...
                    # st.error(f"Warning: Unknown labels found for column '{column}': {unknown_labels}")
                    # if column in label_encoders_3:
                    #     new_data_df[column] = label_encoders_3[column].transform(new_data_df[column])
                    # else:
                    #     raise ValueError(f"Column '{column}' has unknown labels and no backup encoder found.")
                else:
                    new_data_df[column] = encoder.transform(new_data_df[column])
            except (TypeError, ValueError) as e:
                st.error(f"Error encountered for column '{column}': {e}")
                return None

    # Apply scalers
    new_data_df['overall_survival_months'] = scalers['MinMaxScaler()'].transform(new_data_df[['overall_survival_months']])
    new_data_df['death_from_cancer'] = new_data_df['death_from_cancer'].astype(int)
    
    # Make predictions
    new_data_predictions = log_reg_model.predict(new_data_df)
    new_data_probabilities = log_reg_model.predict_proba(new_data_df)[:, 1]
    
    return new_data_predictions[0], new_data_probabilities[0]

# Initialize the objects
load_objects()

# Define the Streamlit UI
st.title("Breast Cancer Survival Prediction")

# User inputs
age_at_diagnosis = st.number_input("Age at Diagnosis", min_value=0.0, value=43.65, step=0.1)
integrative_cluster = st.selectbox("Integrative Cluster", ['4ER+','4ER-','1','2','3','5','6','7','8','9','10'])
oncotree_code = st.selectbox("Oncotree Code", ['IDC','ILC','IMMC','MBC','MDLC'])
overall_survival_months = st.number_input("Overall Survival Months", min_value=0.0, value=0.0, step=0.1)
tumor_other_histologic_subtype = st.selectbox("Tumor Other Histologic Subtype", ['Ductal/NST','Lobular','Medullary','Metaplastic','Mixed','Mucinous','Other','Tubular/ cribriform'])
inferred_menopausal_state = st.selectbox("Inferred Menopausal State", ['Pre','Post'])
pam50_claudin_low_subtype = st.selectbox("PAM50+ Claudin-Low Subtype", ['LumA','Basal','Her2','LumA','LumB','NC','Normal','claudin-low'])
primary_tumor_laterality = st.selectbox("Primary Tumor Laterality", ['Right','Left'])
gene_classifier_subtype = st.selectbox("3-Gene Classifier Subtype", ['ER+/HER2- High Prolif','ER+/HER2- Low Prolif','ER-/HER2-','HER2+'])
type_of_breast_surgery = st.selectbox("Type of Breast Surgery", ['BREAST CONSERVING', 'MASTECTOMY'])
cancer_type_detailed = st.selectbox("Cancer Type Detailed", ['Breast Invasive Ductal Carcinoma','Breast Invasive Lobular Carcinoma','Breast Invasive Mixed Mucinous Carcinoma','Breast Mixed Ductal and Lobular Carcinoma','Metaplastic Breast Cancer'])
death_from_cancer = st.radio("Death from Cancer", [0, 1])
radio_therapy = st.radio("Radio Therapy", [0, 1])

# Prepare the input data
input_data = {
    'age_at_diagnosis': age_at_diagnosis,
    'integrative_cluster': integrative_cluster,
    'oncotree_code': oncotree_code,
    'overall_survival_months': overall_survival_months,
    'tumor_other_histologic_subtype': tumor_other_histologic_subtype,
    'inferred_menopausal_state': inferred_menopausal_state,
    'pam50_+_claudin-low_subtype': pam50_claudin_low_subtype,
    'primary_tumor_laterality': primary_tumor_laterality,
    '3-gene_classifier_subtype': gene_classifier_subtype,
    'type_of_breast_surgery': type_of_breast_surgery,
    'cancer_type_detailed': cancer_type_detailed,
    'death_from_cancer': death_from_cancer,
    'radio_therapy': radio_therapy
}

if st.button("Predict Survival"):
    prediction, probability = predict(input_data)
    if prediction is not None:
        st.write(f"Survival Prediction: {'Yes' if prediction == 0 else 'No'}")
        st.write(f"Survival Probability: {probability * 100:.10f}%")
