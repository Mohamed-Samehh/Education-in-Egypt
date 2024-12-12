import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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

# Step 2: Define high achievement (AVG Score > 80)
data['High_Achiever'] = (data['AVG Score'] > 80).astype(int)  # 1 for high achievers, 0 otherwise

# Step 3: Prepare features (X) and target (y)
X = data[['Parental_Education_Category', 'AVG Score', 'Subject_1', 'Subject_2']]  # Add relevant features
y = data['High_Achiever']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the SVM model
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
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

# Visualize the SVM Decision Boundary (for two features only)
if X.shape[1] == 2:  # Only if we have exactly two features for visualization
    from matplotlib.colors import ListedColormap

    X_train_2d = X_train.iloc[:, :2].values
    y_train_2d = y_train.values

    # Create meshgrid
    x1, x2 = np.meshgrid(
        np.arange(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 0.01),
        np.arange(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 0.01)
    )

    Z = model.predict(np.c_[x1.ravel(), x2.ravel()])
    Z = Z.reshape(x1.shape)

    plt.contourf(x1, x2, Z, alpha=0.8, cmap=ListedColormap(('red', 'green')))
    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_2d, cmap=ListedColormap(('black', 'white')), edgecolor='k')
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
