import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import graphviz

# Load the dataset
data_path = 'D:/BUE- ICS/Fourth year/Semester one/Data Mining/Project/ahh ah/destination_export.csv'
data = pd.read_csv(data_path)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Student year', 'Father Degree', 'Mother Degree', 'Education Type', 'Educational background', 'Age category']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Define performance categories
conditions = [
    (data['AVG Score'] > 80),
    (data['AVG Score'] <= 80) & (data['AVG Score'] > 60),
    (data['AVG Score'] <= 60)
]
choices = ['High', 'Medium', 'Low']
data['Performance'] = np.select(conditions, choices)

# Prepare data for training
X = data.drop(['Student Name', 'First Name', 'Last Name', 'AVG Score', 'Performance'], axis=1)
y = data['Performance']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Visualization
dot_data = export_graphviz(tree_model, out_file=None, feature_names=X_train.columns, class_names=['Low', 'Medium', 'High'],
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("student_performance_tree")  # Saves the tree as a PDF file in the current directory
graph.view()  # Displays the tree

# Evaluate the model
predictions = tree_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
