import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'D:/BUE- ICS/Fourth year/Semester one/Data Mining/Project/ahh ah/cleaned_data.csv'
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

# Step 2: Define high achievement (AVG Score > 80)
data['High_Achiever'] = (data['AVG Score'] > 80).astype(int)  # 1 for high achievers, 0 otherwise

# Step 3: Prepare features (X) and target (y)
X = data[['Parental_Education_Category', 'AVG Score', 'Subject_1', 'Subject_2']]  # Add relevant features
y = data['High_Achiever']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the KNN model
k = 5  # Choose the number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = knn_model.predict(X_test)

# Step 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not High Achiever', 'High Achiever'], yticklabels=['Not High Achiever', 'High Achiever'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
