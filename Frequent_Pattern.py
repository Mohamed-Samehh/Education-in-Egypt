import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data_path = 'D:/BUE- ICS/Fourth year/Semester one/Data Mining/Project/ahh ah/cleanded_data.csv'
data = pd.read_csv(data_path)

# Step 1: Categorize Parental Education
def categorize_parents(father_degree, mother_degree):
    if father_degree != 'None' and mother_degree != 'None':
        return 'Both Educated'
    elif father_degree != 'None' or mother_degree != 'None':
        return 'One Educated'
    else:
        return 'Neither Educated'

data['Parental_Education_Category'] = data.apply(
    lambda row: categorize_parents(row['Father Degree'], row['Mother Degree']), axis=1)

# Step 2: Binarize numerical data into categories
data['AVG_Category'] = pd.cut(data['AVG Score'], bins=[0, 60, 80, 100], labels=['Low AVG', 'Medium AVG', 'High AVG'])
data['Subject_1_Category'] = pd.cut(data['Subject_1'], bins=[0, 60, 80, 100], labels=['Low Subject_1', 'Medium Subject_1', 'High Subject_1'])
data['Subject_2_Category'] = pd.cut(data['Subject_2'], bins=[0, 60, 80, 100], labels=['Low Subject_2', 'Medium Subject_2', 'High Subject_2'])

# Step 3: Prepare transaction-like data
transactions = data[['Parental_Education_Category', 'AVG_Category', 'Subject_1_Category', 'Subject_2_Category']].astype(str).values.tolist()

# Step 4: Calculate item counts
min_support = 0.05  # Minimum support threshold
min_support_count = int(len(transactions) * min_support)  # Absolute count
items = [item for transaction in transactions for item in transaction]
item_counts = pd.Series(items).value_counts()

# Filter items by minimum support
frequent_items = item_counts[item_counts >= min_support_count]
print("Frequent Items:\n", frequent_items)

# Step 5: Generate combinations for frequent itemsets
frequent_itemsets = []
for k in range(1, len(frequent_items) + 1):
    combos = list(combinations(frequent_items.index, k))
    counts = {combo: sum(1 for transaction in transactions if set(combo).issubset(transaction)) for combo in combos}
    filtered = {k: v for k, v in counts.items() if v >= min_support_count}
    frequent_itemsets.extend(filtered.items())

print("\nFrequent Itemsets:")
frequent_itemsets_df = pd.DataFrame(frequent_itemsets, columns=["Itemset", "Support Count"])
frequent_itemsets_df["Support"] = frequent_itemsets_df["Support Count"] / len(transactions)
print(frequent_itemsets_df)

# Step 6: Generate association rules
min_confidence = 0.6  # Minimum confidence threshold
rules = []
for itemset, support_count in frequent_itemsets:
    if len(itemset) < 2:
        continue  # Skip single items
    for k in range(1, len(itemset)):
        for antecedent in combinations(itemset, k):
            consequent = set(itemset) - set(antecedent)
            antecedent_support = sum(1 for transaction in transactions if set(antecedent).issubset(transaction))
            confidence = support_count / antecedent_support
            if confidence >= min_confidence:
                rules.append((set(antecedent), consequent, confidence, support_count / len(transactions)))

rules_df = pd.DataFrame(rules, columns=["Antecedent", "Consequent", "Confidence", "Support"])
rules_df["Antecedent"] = rules_df["Antecedent"].apply(lambda x: ', '.join(x))
rules_df["Consequent"] = rules_df["Consequent"].apply(lambda x: ', '.join(x))
print("\nAssociation Rules:\n", rules_df)

# Step 7: Visualization

# 1. Visualize Frequent Itemsets
top_itemsets = frequent_itemsets_df.nlargest(10, "Support")
plt.figure(figsize=(10, 6))
sns.barplot(data=top_itemsets, x="Support", y=top_itemsets["Itemset"].apply(lambda x: ', '.join(x)), palette="viridis")
plt.title("Top 10 Frequent Itemsets by Support")
plt.xlabel("Support")
plt.ylabel("Itemset")
plt.show()

# 2. Visualize Association Rules
# Scatter plot for Support vs. Confidence with Antecedent-Consequent pairs
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rules_df, x="Support", y="Confidence", size="Confidence", hue="Support", sizes=(20, 200), alpha=0.8, palette="coolwarm")
plt.title("Support vs Confidence of Association Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.legend(title="Support", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

# 3. Visualize Top 10 Rules by Confidence
top_rules = rules_df.nlargest(10, "Confidence")
plt.figure(figsize=(10, 6))
sns.barplot(data=top_rules, x="Confidence", y=top_rules.index, hue="Support", palette="viridis", dodge=False)
plt.title("Top 10 Association Rules by Confidence")
plt.xlabel("Confidence")
plt.ylabel("Rule Index")
plt.legend(title="Support")
plt.show()
