🍽️ Waiter’s Tip Prediction (Machine Learning)

📌 Overview

This project predicts the tip amount a waiter is likely to receive based on various factors such as total bill, number of people, gender, smoking preference, day, and time. The goal is to help understand customer tipping behavior using machine learning techniques.

🎯 Objective
Analyze restaurant dataset to identify patterns affecting tips
Build a regression model to predict tip amount
Improve decision-making using data-driven insights
📊 Dataset

The dataset contains features such as:

total_bill – Total bill amount
tip – Tip given (target variable)
sex – Gender of the customer
smoker – Smoking preference
day – Day of the week
time – Lunch/Dinner
size – Number of people

⚙️ Technologies Used

Python
Pandas & NumPy (data processing)
Matplotlib & Seaborn (visualization)
Scikit-learn (model building)

🔍 Approach

Data Preprocessing
Handled categorical variables using encoding
Checked for missing values and outliers
Exploratory Data Analysis (EDA)
Visualized relationships between bill amount and tip
Analyzed impact of time, day, and group size
Model Building
Implemented Random Forest Regressor
Trained model on processed dataset
Evaluation
Measured performance using R² Score and MAE
Achieved 90%+ prediction accuracy

📈 Results

The model successfully predicts tip amounts with high accuracy
Found that total bill and group size are strong influencing factors
Dinner time generally results in higher tips compared to lunch

🚀 How to Run

# Clone the repository
git clone https://github.com/your-username/waiters-tip-prediction.git

# Navigate to project folder
cd waiters-tip-prediction

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py

📌 Future Improvements

Hyperparameter tuning for better accuracy
Deploy as a web application using Streamlit
Use advanced models like XGBoost


🌐 Streamlit Web Application

To make the model interactive, a Streamlit-based web application was developed. This allows users to input restaurant details and instantly get a predicted tip amount.

✨ Features
User-friendly interface for real-time predictions
Input fields for bill amount, group size, time, day, and other attributes
Instant tip prediction using the trained ML model
Lightweight and easy to run locally

▶️ Run Streamlit App

# Install Streamlit
pip install streamlit

# Run the app
streamlit run app.py

🖥️ How It Works

Users enter input values through the web interface
The trained Random Forest model processes the input
The predicted tip amount is displayed instantly

🚀 Use Case

This application demonstrates how machine learning models can be deployed into real-world interactive tools, making predictions accessible even to non-technical users.

🔮 Future Enhancements

Deploy the app online (Streamlit Cloud / AWS)
Add data visualization dashboard
Improve UI/UX design
Integrate real-time restaurant data

👩‍💻 Author

Muvva Hasini
