import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

#dataset importing
df = pd.read_csv("dataset/clean_it_ticket.csv")
print("dataset shape:", df.shape)
print(df.head())

# print("\nmissing values:")
# print(df.isnull().sum())

#category model creation
#input nd target selection
X = df["ticket_description"]
y_category = df["category"]
y_priority = df["priority"]

print("input shape:", X.shape)
print("category shape:",y_category.shape)
print("priority shape:",y_priority.shape)

#category prediction model
#trainind and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_category, 
    test_size = 0.2, 
    random_state=42, 
    stratify=y_category
)
print("Training data:", X_train.shape)
print("testing data:", X_test.shape)

#tfidf on training data
tfidf = TfidfVectorizer()#object creation
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("Training tfidf shape:", X_train_tfidf.shape)
print("Testing tfidf shape:", X_test_tfidf.shape)

#model training (logistic reg)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("category model traning done")

#prediction
y_predict = model.predict(X_test_tfidf)
print("prediction result:", y_predict[:10])

#model accuracy (evaluation)
accuracy = accuracy_score(y_test, y_predict)
print("model accuracy:",accuracy)
print("model acccuracy in %:",accuracy*100)

#classification report 
print("classificaion report:")#to identify each category (tell how model work on each category)
print(classification_report(y_test, y_predict))
# pricision=prediction
# recall= actual cat. ticket identification
# f1 score= 2 matrix chechup
# support= testing me actual ticket count

#confussion matrix
confusion_mat= confusion_matrix(y_test, y_predict)
print("confusion matrix:")
print(confusion_mat)


#priority model creation
#train test split
X_priority_train, X_priority_test, y_priority_train, y_priority_test = train_test_split(
    X, 
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_priority
)
print("priority tarining data:",X_priority_train.shape)
print("priority testing data:",X_priority_test.shape)

#regration model
y_priority = df["priority"]
priority_model = LogisticRegression(max_iter=1000)
priority_model.fit(
    tfidf.transform(X_priority_train),
    y_priority_train
)
print("Priority model training done")

#priority prediction
priority_predict = priority_model.predict(tfidf.transform(X_priority_test))
print("priority prediction:",priority_predict[:10])

#accuracy
priority_accuracy = accuracy_score(y_priority_test, priority_predict)
print("priority model accuracy:", priority_accuracy)
print("print model accuracy in %:", priority_accuracy*100)


#classification report
print("\npriority calassification report:")
print(classification_report(y_priority_test, priority_predict))

#confusion matrix
priority_confussion_mat = confusion_matrix(y_priority_test, priority_predict)
print("priority confusion matrix:")
print(priority_confussion_mat)

# print("\npriority wise ticket example:")
# for priority in df["priority"].unique():
#     print(f"\n{priority}:")
#     print(df[df["priority"]==priority]["ticket_description"].head(5).to_list())
#     print(pd.crosstab(df["category"],df["priority"]))

# # #additional column work
# # print("\nunique value in each feature:")
# # print(df[[
# #         "category", 
# #         "department", 
# #         "channel",
# #         "status", 
# #         "assigned_agent"
# #     ]].nunique())



#model saving 
joblib.dump(model, "models/category_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
print("category model and itidf vectorizer model saved successfully")

#priority model saving
joblib.dump(priority_model,"models/priority_model.pkl")
print("priority model saved successfully")