import pandas as pd
df = pd.read_csv("dataset/it_ticket_classification_dirty_dataset.csv")
#print(df.to_string())

# data understanding
print("shape of data is:",df.shape)#(5050, 10)
print("\nhead of data:",df.head())
print("\ncolumns in data:",df.columns)

print("\ndata info:",df.info())#none
print("\nmissing values in data:\n",df.isnull().sum())
print("\nduplicates in data:\n",df.duplicated().sum())
print("description of data:\n",df.describe())

#data quality analysis 
missing_values = df.isnull().sum()
missing_percentage = (missing_values/len(df))*100
print("Missing value percentage:\n",missing_percentage)

#dat acleaning
print("\n\n data cleaning:\n")
print(pd.crosstab(df["status"], df["resolution_hours"].isnull()))

print("\n\ncolumn cleaning")
#status
df["status"]=df["status"].str.strip().str.lower()
print(df["status"].value_counts())
#category
df["category"]=df["category"].str.strip().str.lower()
print(df["category"].value_counts())
#priority
df["priority"] = df["priority"].str.strip().str.lower()
print(df["priority"].value_counts())
#department
print(df["department"].value_counts())
#channel
df["channel"]= df["channel"].str.strip().str.lower()
print(df["channel"].value_counts())
#assigned agents
print(df["assigned_agent"].value_counts())
#ticket description
print(df["ticket_description"].isnull().sum())
df.dropna(subset=["ticket_description"],inplace=True)
print(df.shape)
print(df["ticket_description"].isnull().sum())
df["ticket_description"]= df["ticket_description"].str.strip().str.lower()
print(df["ticket_description"].value_counts())
#duplicate removing
print("\nDuplicate rows after cleaning:", df.duplicated().sum())
df = df.drop_duplicates()
print("shape after removing duplicates:",df.shape)
print("Duplicates remaining:", df.duplicated().sum())
#missing value handling
#priority column
print("\nMissing values after cleaning:\n",df.isnull().sum())
print("priority value:\n")
df["priority"]= df["priority"].fillna(df["priority"].mode()[0])
print(df["priority"])
print(df["priority"].isnull().sum())
#department column
print("department :\n")
df["department"] = df["department"].fillna(df["department"].mode()[0])
print(df["department"])
print(df["department"].value_counts())
print(df["department"].isnull().sum())
#assigned agent
print("assigned agent column: \n")
df["assigned_agent"] = df["assigned_agent"].fillna("Unassigned")
print(df["assigned_agent"])
print(df["assigned_agent"].value_counts())
print(df["assigned_agent"].isnull().sum())

#clean file creation
df.to_csv("dataset/clean_it_ticket.csv",index=False)