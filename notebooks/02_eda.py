import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("dataset/clean_it_ticket.csv")

#graph plotting 
#category 
print(df["category"].value_counts())
df["category"].value_counts().plot(kind="bar")
plt.show()
#priority graph
print(df["priority"].value_counts())
df["priority"].value_counts().plot(kind="pie")
plt.show()
#status analysis of ticket creation
print(df["status"].value_counts().sort_values())
df["created_at"] = pd.to_datetime(df["created_at"])
daily_ticket = df.groupby(df["created_at"].dt.date)["ticket_id"].count()
daily_ticket.plot(kind="line")
plt.show()
#resolution hours
df["resolution_hours"].hist(bins=20)
plt.xlabel("hours")
plt.ylabel("No. of ticket")
plt.title("Ticket distribution resolution time")
plt.show()

#inter quartile range(IQR)
Q1=df["resolution_hours"].quantile(0.25)
Q3=df["resolution_hours"].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5 * IQR
upper = Q3+1.5 *IQR

outliers = df[(df["resolution_hours"]<lower)|(df["resolution_hours"]>upper)]
print("total outliers:", len(outliers))

#relationship analysis
print(pd.crosstab(df["category"],df["priority"]))

priority_table = pd.crosstab(df["category"],df["priority"])
sns.heatmap(priority_table, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Category vs Priority")
plt.show()

print("\n\npercentage")
percentage = df.isnull().sum()/len(df)*100
print(percentage)

#missing data of resolution_time
print(df["resolution_hours"].isnull().sum())
print(df.groupby("status")["resolution_hours"].apply(lambda x: x.isnull().sum()))
print(df["status"].value_counts())
print(df.groupby("status")["resolution_hours"].describe())

#outlier in resolution hours
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["resolution_hours"])
plt.title("resolution hours - outlier detection val")
plt.show()

#text data analysis
print("\nticket description sample: ")
print(df["ticket_description"].head(10).to_string(index=False))

print("\ndescription lenght:")
print(df["ticket_description"].str.len().describe())
print("total row: ", len(df))
print("missing description:", df["ticket_description"].isnull().sum())