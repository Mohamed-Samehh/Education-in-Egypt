import pandas as pd

file_path = 'D:/BUE- ICS/Fourth year/Semester one/Data Mining/Project/ahh ah/destination_export.csv' 
data = pd.read_csv(file_path)
print(data.isnull().sum())
data.drop_duplicates(inplace=True)
data.drop(['First Name', 'Last Name'], axis=1, inplace=True)

# The explainationa in the report 

# Save the processed dataset
data.to_csv('cleand_data.csv', index=False)
print("saved ")
