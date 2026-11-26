# IT Ticket Classification Using Python and Machine Learning

## Project Overview
This project uses Machine Learning and Natural Language Processing (NLP) to automatically classify IT support tickets based on their descriptions.

It predicts the ticket category and priority level and provides an interactive dashboard for ticket analysis.

## Objectives
- Automatically classify IT support tickets.
- Predict ticket priority.
- Analyze ticket categories and status.
- Evaluate machine learning model performance.
- Explore and filter the ticket dataset.
- Maintain prediction history using CSV.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- NLP
- TF-IDF Vectorization
- Logistic Regression
- Joblib
- Streamlit

## Machine Learning Models
Two Logistic Regression models are used:
1. Category Classification Model
2. Priority Prediction Model

TF-IDF is used to convert ticket descriptions into numerical features.

## Dashboard Features
- Dashboard Overview
- Ticket Analytics
- Ticket Prediction
- Model Performance Evaluation
- Dataset Explorer
- Prediction History
- CSV Download Option

## Project Structure
```text
IT_Ticket_Classification/
│
├── dataset/
│   ├── clean_it_ticket.csv
│   └── prediction_history.csv
│
├── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 02_eda.py
│   ├── 03_text_preprocessing.py
│   └── 04_model_training.py
│
├── app.py
├── style.css
└── README.md
```

## How to Run the Project

1. Activate the virtual environment.
2. Install the required libraries.
3. Run the following command:

```bash
streamlit run app.py
```
## Output
The application displays predicted ticket categories and priorities along with interactive charts, model evaluation results, dataset exploration, and saved prediction history.
## Conclusion
This project demonstrates how Machine Learning and NLP can help automate IT support ticket classification and improve ticket analysis through an interactive Streamlit dashboard.


