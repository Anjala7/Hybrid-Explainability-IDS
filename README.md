# Hybrid-Explainability-IDS
Intrusion Detection System using MLP with explainability through LIME and SHAP.

Overview
This project implements an Intrusion Detection System (IDS) using a Multi-Layer Perceptron (MLP) classifier, combined with Explainable AI (XAI) techniques LIME and SHAP.
It is based on the UNSW-NB15 dataset and aims to provide human-interpretable explanations for network attack detection.

Features
Preprocessing and encoding of categorical features,
MLP model training and saving,
Performance evaluation (accuracy, classification report, confusion matrix),
LIME explanations for individual predictions,
SHAP explanations for feature importance,
Manual input prediction with live LIME explanation,

📂 Dataset
I use the UNSW-NB15 dataset from Kaggle:

Training set: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15?select=UNSW_NB15_training-set.csv
Testing set: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15?select=UNSW_NB15_testing-set.csv

⚠️ Download these CSV files and place them in the project folder before running the code.

🛠️ Installation & Usage
Clone the repository:

git clone https://github.com/your-Anjala7/Hybrid-Explainability-IDS.git
cd Hybrid-Explainability-IDS

Install dependencies:
pip install pandas numpy shap lime matplotlib seaborn scikit-learn

Run the main script:
python xai.py

📊 Results
Confusion Matrix: confusion_matrix_plot.png
LIME Explanations: lime_explanation_X.html
SHAP Plot: shap_plot.png

License
This project is open-source and available under the MIT License.
