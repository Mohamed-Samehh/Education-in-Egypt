import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'D:/BUE- ICS/Fourth year/Semester one/Data Mining/Project/ahh ah/destination_export.csv'
data = pd.read_csv(data_path)

# Step 1: Categorize parental education
def categorize_parents(father_degree, mother_degree):
    if father_degree != 'None' and mother_degree != 'None':
        return 'Both Educated'
    elif father_degree != 'None' or mother_degree != 'None':
        return 'One Educated'
    else:
        return 'Neither Educated'

data['Parental_Education_Category'] = data.apply(
    lambda row: categorize_parents(row['Father Degree'], row['Mother Degree']), axis=1)

# Encode parental education category
data['Parental_Education_Category'] = data['Parental_Education_Category'].map({
    'Both Educated': 2,
    'One Educated': 1,
    'Neither Educated': 0
})

# Step 2: Define performance categories
def categorize_performance(avg_score):
    if avg_score > 80:
        return 'High'
    elif avg_score >= 60:
        return 'Medium'
    else:
        return 'Low'

data['Performance_Category'] = data['AVG Score'].apply(categorize_performance)

# Encode performance categories for KNN
data['Performance_Category_Encoded'] = data['Performance_Category'].map({
    'Low': 0, 'Medium': 1, 'High': 2
})

# Step 3: Prepare features (X) and target (y)
X = data[['Parental_Education_Category', 'AVG Score', 'Subject_1', 'Subject_2', 'Subject_3', 'Subject_4']]
y = data['Performance_Category_Encoded']

# Step 4: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Check new class distribution
print("Balanced class distribution:\n", pd.Series(y_balanced).value_counts())

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Step 7: Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')  # Use weighted KNN
knn_model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred_knn = knn_model.predict(X_test)

# Step 9: Evaluate the model
print("KNN Accuracy (Balanced Data):", knn_model.score(X_test, y_test))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn, target_names=['Low', 'Medium', 'High']))

# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('KNN Confusion Matrix (Balanced Data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 10: Determine the Optimal Value for k
error_rates = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    error_rates.append(1 - knn.score(X_test, y_test))

# Plot error rates
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), error_rates, marker='o', linestyle='dashed', color='red')
plt.title('Error Rate vs. k Value')
plt.xlabel('k')
plt.ylabel('Error Rate')
plt.show()
