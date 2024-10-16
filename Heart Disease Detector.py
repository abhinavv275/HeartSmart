#!/usr/bin/env python
# coding: utf-8

# # Gradient Boosted Trees
# 
# XGBoost is the primary implementation of gradient boosted trees used in Python. Here, we train a classifier on the Hearts disease dataset to predict whether a give patient has heart disease based on 7 attributes.
# 

# ## Loading in the Data
# First, we import all the necessary libraries as well as the data.

# In[1]:


# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd # For data manipulation
import matplotlib.pyplot as plt 
from scipy.stats import zscore
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
import xgboost
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score,roc_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[2]:


# Ensure plots are displayed inline in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Load the dataset from a CSV file
df = pd.read_csv('Heart_Disease_Prediction.csv')

# Display the first 5 rows of the dataset
df.head()


# In[4]:


# Get the number of rows and columns in the dataset
df.shape


# We have 1295 observations, each with 14 attributes out of which are 13 are features and the last one is the result column.

# In[5]:


# Get summary statistics for numerical columns in the dataset
df.describe()


# ## Exploratory Data Analysis
# 
# This is a necessary step to gauge the quality of the data. First, we need to check the value counts in the target column to make sure they aren't skewed towards one result.
# 

# In[6]:


# Count the occurrences of each unique value in the 'Heart_Disease' column
df.Heart_Disease.value_counts() 


# In[7]:


# Create a bar plot for the 'Heart_Disease' value counts
plt.bar(['1','0'], height=df.Heart_Disease.value_counts())


# # DATA PREPROCESSING 

# Checking for missing values , if any

# In[8]:


# Calculate the number of missing values in each column
missing_values = df.isnull().sum()


# #### Creating a heatmap to visualize missing values

# In[9]:


# Set the figure size for better visibility
plt.figure(figsize=(8, 6))

# Plot a heatmap to visualize missing values in the dataset
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

# Adding title to the heatmap
plt.title('Heatmap of Missing Values')

# Show the plot
plt.show()


# There are no missing values.

# ### Metrics 'Cholesterol', ‘BP’ and 'Max HR' undergoing normalization

# In[10]:


# Initialize the MinMaxScaler for feature scaling
scaler = MinMaxScaler()

# Scale the 'Cholesterol', 'BP', and 'Max HR' columns to a range between 0 and 1
df[['Cholesterol', 'BP', 'Max HR']] = scaler.fit_transform(df[['Cholesterol', 'BP', 'Max HR']])


# In[11]:


# Plot the distribution of Cholesterol with a KDE overlay
plt.figure(figsize=(4, 3))
sns.histplot(df['Cholesterol'], kde=True, color='blue')
plt.title('Distribution of Cholesterol')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of Blood Pressure with a KDE overlay
plt.figure(figsize=(4, 3))
sns.histplot(df['BP'], kde=True, color='green')
plt.title('Distribution of Blood Pressure')
plt.xlabel('Blood Pressure')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of Max HR with a KDE overlay
plt.figure(figsize=(4, 3))
sns.histplot(df['Max HR'], kde=True, color='yellow')
plt.title('Distribution of Max HR')
plt.xlabel('Max HR')
plt.ylabel('Frequency')
plt.show()


# ### Outlier Removal for Cholesterol, BP and Max HR

# In[12]:


# Plot a boxplot for Cholesterol distribution before removing outliers
plt.figure(figsize=(4, 3))
sns.boxplot(x=df['Cholesterol'], color='skyblue')
plt.title('Cholesterol Distribution (Before Removing Outliers)')
plt.xlabel('Cholesterol')
plt.show()

# Calculate Z-scores for the Cholesterol column
df['Cholesterol_z'] = zscore(df['Cholesterol'])

# Filter out outliers based on Z-score
df_filtered = df[(df['Cholesterol_z'].abs() <= 3)]

# Plot a boxplot for Cholesterol distribution after removing outliers
plt.figure(figsize=(4, 3))
sns.boxplot(x=df_filtered['Cholesterol'], color='lightgreen')
plt.title('Cholesterol Distribution (After Removing Outliers)')
plt.xlabel('Cholesterol')
plt.show()


# In[13]:


# Max HR
# Step 1: Boxplot for Max HR before removing outliers
plt.figure(figsize=(4, 3))
sns.boxplot(x=df['Max HR'], color='skyblue')
plt.title('Max HR Distribution (Before Removing Outliers)')
plt.xlabel('Max HR')
plt.show()

# Step 2: Calculate Z-scores for Max HR
df['MaxHR_z'] = zscore(df['Max HR'])

# Step 3: Filter out rows where the Z-score for Max HR is greater than 3 or less than -3
df_filtered_maxhr = df[(df['MaxHR_z'].abs() <= 3)]

# Step 4: Boxplot for Max HR after removing outliers
plt.figure(figsize=(4, 3))
sns.boxplot(x=df_filtered_maxhr['Max HR'], color='lightgreen')
plt.title('Max HR Distribution (After Removing Outliers)')
plt.xlabel('Max HR')
plt.show()


# In[14]:


# BP
# Step 1: Boxplot for BP before removing outliers
plt.figure(figsize=(4, 3))
sns.boxplot(x=df['BP'], color='skyblue')
plt.title('BP Distribution (Before Removing Outliers)')
plt.xlabel('Blood Pressure (BP)')
plt.show()

# Step 2: Calculate Z-scores for BP
df['BP_z'] = zscore(df['BP'])

# Step 3: Filter out rows where the Z-score for BP is greater than 3 or less than -3
df_filtered_bp = df[(df['BP_z'].abs() <= 3)]

# Step 4: Boxplot for BP after removing outliers
plt.figure(figsize=(4, 3))
sns.boxplot(x=df_filtered_bp['BP'], color='lightgreen')
plt.title('BP Distribution (After Removing Outliers)')
plt.xlabel('Blood Pressure (BP)')
plt.show()


# In[15]:


# Filter out rows where the Z-scores for Blood Pressure, Max HR, and Cholesterol are greater than 3 or less than -3
df_filtered = df[(df['BP_z'].abs() <= 3) & (df['MaxHR_z'].abs() <= 3) & (df['Cholesterol_z'].abs() <= 3)]


# In[16]:


# Get the number of rows and columns in the filtered dataset after outlier removal
df_filtered.shape


# ### Feature Selection Using XG Boost

# In[17]:


#Define features and target variable
X = df_filtered[['Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 
         'EKG results', 'Max HR', 'Exercise angina', 
         'ST depression', 'Slope of ST', 'Number of vessels fluro', 
         'Thallium','Age','Sex']]
y = df_filtered['Heart_Disease']  

# Train the XGBoost model
model = XGBClassifier()
model.fit(X, y)

# Plot feature importance
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight', max_num_features=14, title='Feature Importance for Heart Disease Prediction', xlabel='F score', ylabel='Features')
plt.show()


# We have got our most 7 importrant features :- Cholestrol, Age, Max HR, ST depression, BP, Chest pain type, Number of vessels fluro
# 

# # Model Training
# 

# ### XG Boost Classifier
# 

# In[18]:


df_filtered.describe()


# In[19]:


# Step 2: Select only the most important features
X = df_filtered[['Cholesterol', 'Max HR', 'Age', 'ST depression', 'BP', 'Number of vessels fluro', 'Chest pain type']]
y = df_filtered['Heart_Disease']  # The target variable

# Step 3: Split the dataset (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Step 4: Initialize the XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (Heart Disease)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Step 8: Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")

# Step 9: Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(5, 3))
plt.plot(fpr, tpr, color='orange', label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# ### Ada Boost Classifier
# 

# In[20]:


# Feature selection
X = df_filtered[['Cholesterol', 'Max HR', 'Age', 'ST depression', 'BP', 'Number of vessels fluro', 'Chest pain type']]
y = df_filtered['Heart_Disease']  # Target variable

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the AdaBoost Classifier
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
ada_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ada_model.predict(X_test)
y_pred_proba = ada_model.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f'AUC: {auc:.4f}')

# Plot the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(5, 3))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.title('Receiver Operating Characteristic (ROC) Curve - AdaBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# ### Comparing XG Boost Classsifier and Ada Boost classifier

# In[21]:


# Features and target variable
features = ['Cholesterol', 'Max HR', 'Age', 'ST depression', 'BP', 'Number of vessels fluro', 'Chest pain type']
X = df_filtered[features]
y = df_filtered['Heart_Disease']

# Split the dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize models
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, use_label_encoder=False)

# Train the AdaBoost model
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Print classification report for both models
print("Classification Report for AdaBoost:")
print(classification_report(y_test, y_pred_ada))

print("\nClassification Report for XGBoost:")
print(classification_report(y_test, y_pred_xgb))


# In[22]:


# Compute AUC for both models
roc_auc_ada = roc_auc_score(y_test, ada_model.predict_proba(X_test)[:, 1])
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

print(f"AdaBoost ROC AUC: {roc_auc_ada}")
print(f"XGBoost ROC AUC: {roc_auc_xgb}")


# In[23]:


# Plot confusion matrix for AdaBoost
plt.figure(figsize=(4,4))
conf_matrix_ada = confusion_matrix(y_test, y_pred_ada)
disp_ada = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_ada)
disp_ada.plot(cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix: AdaBoost')
plt.show()


# In[24]:


plt.figure(figsize=(4,4))
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_xgb)
disp_xgb.plot(cmap='Greens', ax=plt.gca())
plt.title('Confusion Matrix: XGBoost')
plt.show()



# In[25]:


fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_model.predict_proba(X_test)[:, 1])
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(6, 4))
plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC = {roc_auc_ada:.2f})', color='blue')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# # Tuning

# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt


data = pd.read_csv('Heart_Disease_Prediction.csv')  


features = [
    'Chest pain type', 'BP', 'Cholesterol', 
    'EKG results', 'Max HR',
    'ST depression',  'Number of vessels fluro',
    'Thallium', 'Age', 'Sex'
]
target = 'Heart_Disease'


X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tuned_model = xgb.XGBClassifier(
    gamma=3,
    learning_rate=0.05,
    eval_metric='error' 
)
tuned_model.fit(X_train, y_train)


train_accuracy = accuracy_score(tuned_model.predict(X_train), y_train)
print("Training accuracy: ", train_accuracy)

y_pred_tuned = tuned_model.predict(X_test)


test_accuracy = accuracy_score(y_pred_tuned, y_test)
print("Test accuracy: ", test_accuracy)



# Calculate overall accuracy by combining train and test accuracies
overall_accuracy = np.mean([train_accuracy, test_accuracy])
print("Overall accuracy: ", overall_accuracy)


# In[27]:


y_proba = tuned_model.predict_proba(X_test)[:, 1]


fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

