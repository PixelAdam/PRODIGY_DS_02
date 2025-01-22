import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_data=pd.read_csv(r"C:\Users\ADAM\Downloads\titanic\train.csv")

# Check for missing values in the dataset "train"
print(train_data.isnull().sum())

# Checking the distribution of Age so we can see how to clean missing Age values
# Histogram : Age Distribution
sns.histplot(train_data['Age'],bins=30,kde=True)
plt.title('Age Distribution')
plt.show()

# Checking details of missing "Embarked" rows
print(train_data[train_data['Embarked'].isnull()])

# Checking embarkation points of similar passengers that have Pclass=1
similar_passengers=train_data[train_data['Pclass']==1]
print(similar_passengers[['Embarked','Pclass']])

# Using median to fill the missing Age values
train_data['Age']=train_data['Age'].fillna(train_data['Age'].median())

# Filling missing Cabin numbers with "unknown" cabin number
train_data['Cabin']=train_data['Cabin'].fillna('Unknown')

# Filling the missing "Embarked" values with "S"(Southampton) based on the other embarkation that have Pclass=1
train_data['Embarked']=train_data['Embarked'].fillna('S')

# Checking if there are now any missing Age values and Cabin numbers
print("The missing \"Age\", values are now :",train_data['Age'].isnull().sum())
print("The missing \"Cabin\" numbers are now :",train_data['Cabin'].isnull().sum())
print("The missing \"Embarked\" letters are now :",train_data['Embarked'].isnull().sum())

# Verify the dataset
print(train_data.isnull().sum())

# Save the cleaned dataset
train_data.to_csv('cleaned_titanic_dataset.csv',index=False)

# View some rows of the saved cleaned dataset
cleaned_data=pd.read_csv('cleaned_titanic_dataset.csv')
print(cleaned_data.head())

# Performing Exploratory Data Analysis (EDA)

# Since we already performed a visualization of the Age Distribution, now we will visualize categorical variables like "Pclass"
sns.countplot(x='Pclass',data=train_data)
plt.title('Class Distribution')
plt.show()

# Analyzing relationships : Comparing "Survived" against other variables ("Pclass","Sex")
sns.barplot(x='Pclass',y='Survived',data=train_data)
plt.title('Survival Rate by Class')
plt.show()

sns.barplot(x='Sex',y='Survived',data=train_data)
plt.title('Survival Rate by Gender')
plt.show()
