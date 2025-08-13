import os
import pandas as pd
import numpy as np
import pickle
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Load dataset
train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")

# ✅ Drop unnecessary columns
columns_to_drop = ['id']
train_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
test_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# ✅ Identify categorical columns & encode them
categorical_cols = train_df.select_dtypes(include=['object']).columns
encoders = {}

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    train_df[col] = encoders[col].fit_transform(train_df[col])

    test_df[col] = test_df[col].apply(lambda x: x if x in encoders[col].classes_ else "Unknown")
    encoders[col].classes_ = np.append(encoders[col].classes_, "Unknown")
    test_df[col] = encoders[col].transform(test_df[col])

# ✅ Save encoders
with open("encoders.pkl", "wb") as file:
    pickle.dump(encoders, file)

# ✅ Split features & target
X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

# ✅ Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Save the scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# ✅ Load or train model
model_filename = "mlp_model.pkl"

if os.path.exists(model_filename):
    print("\n✅ Loading saved MLP model...")
    with open(model_filename, "rb") as file:
        model = pickle.load(file)
else:
    print("\n⚠️ No saved model found! Training a new MLP model...")
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    with open(model_filename, "wb") as file:
        pickle.dump(model, file)

print("\n✅ Model Ready.")

# ✅ Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model Accuracy:", accuracy)
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Confusion Matrix Heatmap ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (0)', 'Attack (1)'],
            yticklabels=['Normal (0)', 'Attack (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_plot.png')
plt.show()

# === Manual Prediction & LIME explanation ===
def manual_prediction_with_lime():
    try:
        with open("mlp_model.pkl", "rb") as file:
            model = pickle.load(file)
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        with open("encoders.pkl", "rb") as file:
            encoders = pickle.load(file)

        feature_names = test_df.columns[:-1]
        X_train_raw = scaler.inverse_transform(X_train)  # For LIME initialization

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            mode="classification",
            feature_names=feature_names,
            discretize_continuous=False
        )

        while True:
            print("\n🔹 Enter feature values for prediction:")
            user_input = []

            for feature_name in feature_names:
                if feature_name in encoders:
                    valid_values = list(encoders[feature_name].classes_)
                    print(f"🔸 {feature_name} (Categorical) - Options: {valid_values}")
                    while True:
                        value = input(f"Enter value for {feature_name}: ").strip()
                        if value in valid_values:
                            user_input.append(encoders[feature_name].transform([value])[0])
                            break
                        else:
                            print(f"❌ Invalid choice! Try again.")
                else:
                    while True:
                        try:
                            value = float(input(f"Enter value for {feature_name}: "))
                            user_input.append(value)
                            break
                        except ValueError:
                            print("❌ Invalid input! Please enter a number.")

            input_data = np.array(user_input).reshape(1, -1)
            scaled_input = scaler.transform(input_data)

            prediction = model.predict(scaled_input)
            confidence = model.predict_proba(scaled_input)

            result_text = "✅ Normal Traffic" if prediction[0] == 0 else "🚨 Intrusion Detected"
            print(f"\n✅ Intrusion Detection Result: {result_text}")
            print(f"✅ Confidence Score: {confidence}")

            # 🔍 Generate LIME explanation for this input
            exp = explainer_lime.explain_instance(
                scaled_input[0],
                model.predict_proba
            )
            exp.save_to_file("lime_manual_input.html")
            print("\n✅ LIME explanation for manual input saved as 'lime_manual_input.html'.")

            cont = input("\n🔄 Do you want to enter another tuple? (yes/no): ").strip().lower()
            if cont != 'yes':
                break

    except Exception as e:
        print("\n❌ Error in Manual Prediction:", str(e))

# === Run manual input before others ===
manual_prediction_with_lime()

# === Random LIME Explanations ===
print("\n✅ Generating LIME explanation for random test samples...")
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    mode="classification",
    feature_names=test_df.columns[:-1],
    discretize_continuous=False
)

num_samples = 5
for i in range(num_samples):
    idx = np.random.randint(0, X_test.shape[0])
    exp = explainer_lime.explain_instance(X_test[idx], model.predict_proba)
    exp.save_to_file(f"lime_explanation_{i+1}.html")
    print(f"✅ LIME explanation {i+1} saved as 'lime_explanation_{i+1}.html'.")

# === SHAP Explanation ===
print("\n✅ Generating SHAP explanation...")
explainer_shap = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
shap_values = explainer_shap.shap_values(shap.sample(X_test, 50))
shap.summary_plot(shap_values, X_test[:50])
plt.show()
print("\n✅ SHAP explanation complete.")




# Uncomment below to explain a specific instance using LIME
# # explain_instance(0)

# Manual Intrusion Detection
# def predict_intrusion():
    # user_input = []
    # for feature in train_df.columns[:-1]:  # Excluding the target column
        # value = input(f"Enter value for {feature}: ")
        # try:
            # value = float(value)
        # except ValueError:
            # value = encoder.transform([value])[0] if feature in categorical_cols else 0
        # user_input.append(value)

    # user_input = np.array(user_input).reshape(1, -1)
    # user_input = scaler.transform(user_input)
    # prediction = model.predict(user_input)[0]
    # print(f"Predicted Intrusion Class: {prediction}")

# Uncomment below to enable manual prediction
# predict_intrusion()

