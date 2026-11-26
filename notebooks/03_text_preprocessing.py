import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

df= pd.read_csv("dataset/clean_it_ticket.csv")
print(df["ticket_description"].head())
#lower case convert
df["ticket_description"]= df["ticket_description"].str.lower()
print(df["ticket_description"].head())
#special char remove
df["ticket_description"]=df["ticket_description"].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
print(df["ticket_description"].head())
#extra space remove
df["ticket_description"]=df["ticket_description"].str.replace(r"\s+", " ", regex=True).str.strip()
print(df["ticket_description"].head())

#stopword remove(text preprocessing)

stop_words = set(stopwords.words("english"))
stop_words = stop_words-{"not", "no", "nor", "never"}
df["ticket_description"] = df["ticket_description"].apply(
    lambda text: " ".join(
        word for word in text.split() if word not in stop_words)) 
print(df["ticket_description"].head())

#tfidf
tfidf = TfidfVectorizer()
print(tfidf)
print("\nAfter stopword removal: ")
print(df["ticket_description"].head(10).to_list())
X = tfidf.fit_transform(df["ticket_description"])
print("tf-idf shape: ",X.shape)
print(X.toarray()[:2])
print(tfidf.get_feature_names_out())