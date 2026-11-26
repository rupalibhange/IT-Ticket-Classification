
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# page confragation
st.set_page_config(
    page_title="IT TICKET CLASSIFICATION",
    page_icon="🎫",
    layout="wide"
)
# load css
with open("style.css", "r") as file:
    css = file.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# navigation bar
st.sidebar.title("🎫 IT TICKET AI")
st.sidebar.caption("Ticket Management System")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Analytics",
        "Prediction",
        "Model Performance",
        "Dataset Explorer",
        "About"
    ]
)

# datasrt & model loading
df = pd.read_csv("dataset/clean_it_ticket.csv")
category_model = joblib.load("models/category_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# header
def show_header():
    header_col, status_col = st.columns([5, 1])
    with header_col:
        st.title("🎫 IT TICKET CLASSIFICATION")
        st.subheader(
            "Machine Learning Based IT Ticket Analysis and Prediction"
        )
    with status_col:
        st.markdown(
            """
            <div class="system-status">
                🟢 System Ready
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("---")
# kpi card
def show_kpis():
    st.subheader("📌 Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets", len(df))
    col2.metric("Categories", df["category"].nunique())
    col3.metric("Priority Levels", df["priority"].nunique())
    col4.metric("Departments", df["department"].nunique())
# chart
def show_charts():
    st.subheader("📊 Ticket Analytics")
    chart1, chart2, chart3 = st.columns([1.4, 1, 1])
    # cat - wise ticket
    with chart1:
        st.markdown("**📊 Category-wise Tickets**")
        category_counts = df["category"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            category_counts.index,
            category_counts.values,
            color="#24476B",
            edgecolor="#17365D",
            width=0.6
        )
        ax.set_xlabel("Category")
        ax.set_ylabel("Number of Tickets")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # priority destribution
    with chart2:
        st.markdown("**🎯 Priority Distribution**")
        priority_counts = df["priority"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(
            priority_counts.values,
            labels=priority_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={
                "edgecolor": "white",
                "linewidth": 2
            }
        )
        ax2.axis("equal")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ticket status
    with chart3:
        st.markdown("**📌 Ticket Status**")
        status_counts = df["status"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={
                "width": 0.4,
                "edgecolor": "white",
                "linewidth": 2
            }
        )
        ax3.axis("equal")
        fig3.tight_layout()

        st.pyplot(fig3)
        plt.close(fig3)
# ticket prediction
def show_prediction():
    st.subheader("✨ Ticket Prediction")
    with st.container(border=True):
        with st.form("prediction_form"):
            ticket_text = st.text_input(
                "Ticket Description",
                placeholder="Enter your IT ticket description"
            )
            predict = st.form_submit_button(
                "🔍 Predict Ticket"
            )
        if predict:
            if ticket_text.strip() == "":
                st.warning("Please enter a ticket description.")
            else:
                ticket_tfidf = tfidf.transform([ticket_text])
                predicted_category = category_model.predict(
                    ticket_tfidf
                )[0]
                predicted_priority = priority_model.predict(
                    ticket_tfidf
                )[0]
                # csv prediction saving
                prediction_data = pd.DataFrame([{
                    "date_time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "ticket_description": ticket_text,
                    "predicted_category": predicted_category,
                    "predicted_priority": predicted_priority
                }])
                file_path = "dataset/prediction_history.csv"
                os.makedirs("dataset", exist_ok=True)
                prediction_data.to_csv(
                    file_path,
                    mode="a",
                    header=not os.path.exists(file_path),
                    index=False
                )
                st.success(
                    "Prediction completed and saved successfully!"
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.success(
                        f"Category: {predicted_category}"
                    )
                with col2:
                    st.success(
                        f"Priority: {predicted_priority}"
                    )
# model performance
def show_model_performance():
    st.subheader("📈 Model Performance")
    X = df["ticket_description"]
    y_category = df["category"]
    y_priority = df["priority"]
    # category test data
    _, X_test, _, y_test = train_test_split(
        X,
        y_category,
        test_size=0.2,
        random_state=42,
        stratify=y_category
    )
    # priority test data
    _, X_priority_test, _, y_priority_test = train_test_split(
        X,
        y_priority,
        test_size=0.2,
        random_state=42,
        stratify=y_priority
    )
    # cat prediction
    X_test_tfidf = tfidf.transform(X_test)
    category_predictions = category_model.predict(
        X_test_tfidf
    )
    category_accuracy = accuracy_score(
        y_test,
        category_predictions
    )
    # priority prediction
    X_priority_test_tfidf = tfidf.transform(
        X_priority_test
    )
    priority_predictions = priority_model.predict(
        X_priority_test_tfidf
    )
    priority_accuracy = accuracy_score(
        y_priority_test,
        priority_predictions
    )
    # accuracy card
    col1, col2 = st.columns(2)
    col1.metric(
        "Category Model Accuracy",
        f"{category_accuracy * 100:.2f}%"
    )
    col2.metric(
        "Priority Model Accuracy",
        f"{priority_accuracy * 100:.2f}%"
    )
    # classification report
    with st.expander("View Classification Reports"):
        category_report = classification_report(
            y_test,
            category_predictions,
            output_dict=True,
            zero_division=0
        )
        priority_report = classification_report(
            y_priority_test,
            priority_predictions,
            output_dict=True,
            zero_division=0
        )
        st.write("Category Model Report")
        st.dataframe(
            pd.DataFrame(category_report).transpose(),
            use_container_width=True
        )
        st.write("Priority Model Report")
        st.dataframe(
            pd.DataFrame(priority_report).transpose(),
            use_container_width=True
        )
    # confusion matrix
    with st.expander("View Confusion Matrices"):
        cm1, cm2 = st.columns(2)
        with cm1:
            st.write("Category Model")
            category_cm = confusion_matrix(
                y_test,
                category_predictions,
                labels=category_model.classes_
            )
            st.dataframe(
                pd.DataFrame(
                    category_cm,
                    index=category_model.classes_,
                    columns=category_model.classes_
                )
            )
        with cm2:
            st.write("Priority Model")
            priority_cm = confusion_matrix(
                y_priority_test,
                priority_predictions,
                labels=priority_model.classes_
            )
            st.dataframe(
                pd.DataFrame(
                    priority_cm,
                    index=priority_model.classes_,
                    columns=priority_model.classes_
                )
            )
# dataset explorer
def show_dataset_explorer():
    st.subheader("🔎 Dataset Explorer")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_category = st.selectbox(
                "Select Category",
                ["All"] + sorted(
                    df["category"].dropna().unique().tolist()
                )
            )
        with col2:
            selected_priority = st.selectbox(
                "Select Priority",
                ["All"] + sorted(
                    df["priority"].dropna().unique().tolist()
                )
            )
        with col3:
            selected_status = st.selectbox(
                "Select Status",
                ["All"] + sorted(
                    df["status"].dropna().unique().tolist()
                )
            )
        filtered_df = df.copy()
        if selected_category != "All":
            filtered_df = filtered_df[
                filtered_df["category"] == selected_category
            ]
        if selected_priority != "All":
            filtered_df = filtered_df[
                filtered_df["priority"] == selected_priority
            ]
        if selected_status != "All":
            filtered_df = filtered_df[
                filtered_df["status"] == selected_status
            ]
        st.write("Total Records:", len(filtered_df))
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=300
        )
# prediction history
def show_prediction_history():
    st.subheader("📂 Prediction History")
    file_path = "dataset/prediction_history.csv"
    if os.path.exists(file_path):
        history_df = pd.read_csv(file_path)
        st.write("Total Predictions:", len(history_df))
        st.dataframe(
            history_df,
            use_container_width=True
        )
        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")
        st.download_button(
            "⬇️ Download Prediction History",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info(
            "No prediction history available yet. "
            "Make your first ticket prediction."
        )
# dashboard page
if page == "Dashboard":
    show_header()
    show_kpis()
    st.markdown("---")
    show_charts()
    st.markdown("---")
    show_prediction()
    st.markdown("---")
    left_col, right_col = st.columns(2)
    with left_col:
        with st.container(border=True):
            show_model_performance()
    with right_col:
        with st.container(border=True):
            show_dataset_explorer()
    st.markdown("---")
    show_prediction_history()
# analysis page
elif page == "Analytics":
    show_header()
    show_charts()
# prediction page
elif page == "Prediction":
    show_header()
    show_prediction()
    show_prediction_history()
# model performance page
elif page == "Model Performance":
    show_header()
    show_model_performance()
# dataset explorer page
elif page == "Dataset Explorer":
    show_header()
    show_dataset_explorer()
# about page
elif page == "About":
    show_header()
    st.title("About This Project")
    st.write(
        """
        IT Ticket Classification is a Machine Learning based project
        designed to automatically classify IT support tickets.
        **Technologies Used:**
        - Python
        - Pandas and NumPy
        - Matplotlib
        - Scikit-learn
        - NLP and TF-IDF
        - Logistic Regression
        - Joblib
        - Streamlit
        **Project Objectives:**
        - Predict ticket categories.
        - Predict ticket priorities.
        - Analyze ticket distributions.
        - Evaluate Machine Learning models.
        - Explore the IT ticket dataset.
        - Save prediction history using CSV.
        """
    )
