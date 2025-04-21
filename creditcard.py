# %% [markdown]
# ## Credit Card Fraud Detection using Machine learning 
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %% [markdown]
# ### Loading dataset into data frame

# %%
# importing data

transaction_dataset= pd.read_csv("creditcard.csv")
transaction_dataset.head(10)

# %% [markdown]
# 

# %%
# last 5 rows

transaction_dataset.tail()

# %% [markdown]
# ### Data Analysis

# %%
# check shape of dataset

transaction_dataset.shape

# dataset has 73377 rows and 31 columns

# %%
# dataset information 

transaction_dataset.info()

# %%
# dataset datatype

transaction_dataset.dtypes

# %%
# mathematical values of dataset

transaction_dataset.describe()

# %%
# finding null values in dataset

transaction_dataset.isnull().sum()

# %%
# total null values in all rows and columns

transaction_dataset.isnull().sum().sum()

# hence there are total 28 null values

# %%
# distribution of legit transaction and fraudulent transaction

transaction_dataset['Class'].value_counts()

# data is highly unbalanced 

# %% [markdown]
# ### Seperating data for analysis
#   - 0 : Normal transaction
#   - 1 : Fraudulent transaction

# %%
legit = transaction_dataset[transaction_dataset.Class == 0]
fraud = transaction_dataset[transaction_dataset.Class == 1]

# %%
print("Shape of legit : ", legit.shape)
print("Shape of fraud : ", fraud.shape)

# %%
# measures

legit.Amount.describe()

# %%
fraud.Amount.describe()
# fraud transaction description

# %%
# comparing the values for both transaction 

transaction_dataset.groupby('Class').mean()

# %% [markdown]
# ### Under-Sampling 
# 
# - build a sample dataset having similar distribution of normal and fraudulent transactions.
# - number of fraudulent transaction is = 492

# %%
ligit_sample = transaction_dataset.sample(n = 492)

# %% [markdown]
# #### Concatinating two samplings

# %%
new_transaction_dataset2 = pd.concat([ligit_sample, fraud], axis = 0)

# %%
new_transaction_dataset2.head()

# %%
new_transaction_dataset2.tail()

# %%
new_transaction_dataset2['Class'].value_counts()

# %%
new_transaction_dataset2.groupby('Class').mean()

# %% [markdown]
# ### Data Visualization

# %%
plt.figure(figsize = (20,11))
# heatmap size in ration 16:9

sns.heatmap(new_transaction_dataset2.corr(), annot = True, cmap = 'coolwarm')
# heatmap parameters

plt.title("Heatmap for correlation matrix for credit card data ", fontsize = 22)
plt.show()

# %%
sns.countplot(new_transaction_dataset2.Class)

# %% [markdown]
# ### Splitting the data into features and Targets

# %%
X = new_transaction_dataset2.drop(columns = 'Class', axis = 1)
Y = new_transaction_dataset2['Class']

# %%
X

# %%
Y

# %% [markdown]
# ### Splitting the data into training and testing data
# 

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 2)

# %%
print("Shape of X_train ", X_train.shape)
print("Shape of X_test ", X_test.shape)
print("Shape of Y_train ", Y_train.shape)
print("Shape of Y_test ", Y_test.shape)


# %% [markdown]
# ### Model Training 

# %%
model = LogisticRegression()
model.fit(X_train, Y_train)

# %% [markdown]
# ## Model Evaluation
# 
# - Accuracy Score

# %%
# accuracy on training data 

X_train_prediction = model.predict(X_train)
traning_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# %%
print("Accuracy on Training data ",traning_data_accuracy)

# %%
# accuracy on testing data 

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# %%
print("Accuracy on Training data ",test_data_accuracy)

# %%
from sklearn.metrics import  precision_score, recall_score, f1_score
Y_pred = model.predict(X_test)

precision = precision_score(Y_test, Y_pred)
print("Precision:", precision)



# %%



