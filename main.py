# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

import pandas as pd

df = pd.read_csv('/content/parkinsons.csv')
display(df.head())

print(df.columns)

input_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']
output_feature = 'status'

X = df[input_features]
y = df[output_feature]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled input features (first 5 rows):\n", X_scaled[:5])

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the Support Vector Machine classifier
model = SVC(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("Model training complete.")

# 6. Test the accuracy:
# Evaluate the model's accuracy on the test set. Ensure that the accuracy is at least 0.8.

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

if accuracy >= 0.8:
    print("Accuracy target of 0.8 met!")
else:
    print("Accuracy target of 0.8 not met. Consider adjusting features or model parameters.")

import joblib

joblib.dump(model, 'spongebob.joblib')
