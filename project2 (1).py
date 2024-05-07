import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv('combine_data.csv')
print(data.head())
print(data.dtypes)

# Handle missing values using KNN Imputer for numerical data
imputer = KNNImputer(n_neighbors=5)
columns_to_impute = ['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle']
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

# Handle missing values for Round and Pick columns
data['Undrafted'] = (data['Round'].isna() & data['Pick'].isna()).astype(int)
data.loc[data['Undrafted'] == 1, 'Undrafted'] = 'Yes'
data.loc[data['Undrafted'] == 0, 'Undrafted'] = 'No'

# Basic statistical analysis
print(data.describe())

# Visualize relationships between features
sns.pairplot(data)
plt.show()

# Generate and display confusion matrix for logistic regression
log_reg_cm = confusion_matrix(y_test, log_reg_pred)
log_reg_cm_display = ConfusionMatrixDisplay(confusion_matrix=log_reg_cm, display_labels=['No', 'Yes'])
log_reg_cm_display.plot()
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Generate and display confusion matrix for random forest
rand_forest_cm = confusion_matrix(y_test, rand_forest_pred)
rand_forest_cm_display = ConfusionMatrixDisplay(confusion_matrix=rand_forest_cm, display_labels=['No', 'Yes'])
rand_forest_cm_display.plot()
plt.title('Random Forest Confusion Matrix')
plt.show()


# Correlation analysis
corr_matrix = data[['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Grouped analysis by position
grouped_data = data.groupby('Pos')[['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle']].mean()
print(grouped_data)

# Visualization by position
plt.figure(figsize=(12, 8))
for pos, group in data.groupby('Pos'):
    plt.scatter(group['Ht'], group['Wt'], label=pos, alpha=0.5)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height vs Weight by Position')
plt.legend()
plt.show()

# Split the data into training and test sets
X = data[['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle']]
y = data['Undrafted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression and random forest classifiers
log_reg = LogisticRegression(max_iter=1000)
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)

log_reg.fit(X_train, y_train)
rand_forest.fit(X_train, y_train)

# Evaluate logistic regression and random forest classifiers on the test set
log_reg_pred = log_reg.predict(X_test)
rand_forest_pred = rand_forest.predict(X_test)

log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
rand_forest_accuracy = accuracy_score(y_test, rand_forest_pred)

print(f"Logistic Regression Accuracy: {log_reg_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {rand_forest_accuracy * 100:.2f}%")

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, log_reg_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rand_forest_pred))

# Function to predict the likelihood of being drafted as a percentage
def predict_draft_likelihood():
    # Get user input and check for missing values
    user_data = {}
    for feature in ['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle']:
        user_input = input(f"Enter {feature} (leave blank if unknown): ")
        if user_input == '':
            user_data[feature] = np.nan  # Use NaN for missing values
        else:
            user_data[feature] = float(user_input)  # Convert input to float

    # Convert user data to a dataframe
    user_data_df = pd.DataFrame([user_data])

    # Handle missing user input using the KNN imputer
    user_data_imputed = imputer.transform(user_data_df)

    # Predict the likelihood of being drafted using logistic regression
    log_reg_prob = log_reg.predict_proba(user_data_imputed)
    log_reg_probability = log_reg_prob[0][1]  # Probability of being drafted based on logistic regression

    # Predict the likelihood of being drafted using random forest
    rand_forest_prob = rand_forest.predict_proba(user_data_imputed)
    rand_forest_probability = rand_forest_prob[0][1]  # Probability of being drafted based on random forest

    # Convert probabilities to percentages and print the results
    log_reg_percentage = log_reg_probability * 100
    rand_forest_percentage = rand_forest_probability * 100

    print(f"Based on your input, your chance of being drafted is as follows:")
    print(f"Logistic Regression: {log_reg_percentage:.2f}%")
    print(f"Random Forest: {rand_forest_percentage:.2f}%")

# Run the user interface function
predict_draft_likelihood()
