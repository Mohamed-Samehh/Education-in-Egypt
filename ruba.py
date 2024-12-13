import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = 'cleaned_data.csv' 
data = pd.read_csv(file_path)

# Handle missing values (fill with median for simplicity)
data.fillna(data.median(numeric_only=True), inplace=True)

# Define thresholds for performance levels
def categorize_performance(avg_score):
    if avg_score < 50:
        return 'Low'
    elif 50 <= avg_score < 75:
        return 'Medium'
    else:
        return 'High'

# Create a new target column 'Performance Level'
data['Performance Level'] = data['AVG Score'].apply(categorize_performance)

# Encode the target variable
label_encoder = LabelEncoder()
data['Performance Level'] = label_encoder.fit_transform(data['Performance Level'])

# Select features and target
X = data.drop(columns=['Student Name', 'Predicted_Graduation_Year', 'Education Type', 'Performance Level'])  # Drop unnecessary columns
y = data['Performance Level']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42, max_depth=5)  # Adjust max_depth as needed

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Feature importance
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index=X_train.columns,
                                   columns=['Importance']).sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# Optional: Visualize the decision tree
from sklearn.tree import export_text
print("Decision Tree Rules:")
print(export_text(model, feature_names=list(X.columns)))