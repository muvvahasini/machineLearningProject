import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import uniform

# Streamlit title
st.title("waiter's Tip Prediction App")

# Load the dataset
uploaded_file = st.file_uploader("Upload the dataset (tips.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Features and target
    X = df.drop(columns=['tip'])
    y = df['tip']

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline: scaling + Gradient Boosting
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # StandardScaler for feature scaling
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    # Define hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'regressor__n_estimators': [100, 150, 200, 250],
        'regressor__max_depth': [3, 6, 10, 15],
        'regressor__learning_rate': uniform(0.01, 0.2),
        'regressor__subsample': uniform(0.7, 0.3),
        'regressor__min_samples_split': [2, 5, 10]
    }

    # Randomized search for the best model
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=3, n_iter=50, scoring='r2', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Best model evaluation
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate accuracy as a percentage
    accuracy = (1 - mae / y_test.mean()) * 100

    # Show Evaluation Metrics
    st.write("### Best Model Parameters:")
    st.write(random_search.best_params_)
    st.write("### Model Performance Metrics")
    st.metric(label="Model Accuracy", value=f"{accuracy:.2f}%")
    st.metric(label="R-squared (R²)", value=f"{r2:.2f}")
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")

    # Visualization: Actual vs Predicted Tips
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.6)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax1.set_xlabel("Actual Tips")
    ax1.set_ylabel("Predicted Tips")
    ax1.set_title("Actual vs Predicted Tips")
    st.pyplot(fig1)

    # Feature Importance Visualization
    feature_importances = best_model.named_steps['regressor'].feature_importances_
    features = X_train.columns

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importances)
    ax2.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    ax2.set_yticks(np.arange(len(sorted_idx)))
    ax2.set_yticklabels(features[sorted_idx])
    ax2.set_xlabel("Feature Importance")
    ax2.set_title("Feature Importances from Gradient Boosting")
    st.pyplot(fig2)

    # Prediction Example
    st.write("### Predict Tips for New Data")
    total_bill = st.number_input("Total Bill", min_value=0.0, value=50.0)
    size = st.number_input("Party Size", min_value=1, value=3)
    sex_male = st.selectbox("Is Male?", [1, 0])
    smoker_yes = st.selectbox("Is Smoker?", [1, 0])
    day_fri = st.selectbox("Is Day Friday?", [1, 0])
    day_sat = st.selectbox("Is Day Saturday?", [1, 0])
    day_sun = st.selectbox("Is Day Sunday?", [1, 0])
    time_dinner = st.selectbox("Is Dinner?", [1, 0])

    new_data = pd.DataFrame({
        'total_bill': [total_bill], 
        'size': [size], 
        'sex_Male': [sex_male], 
        'smoker_Yes': [smoker_yes],
        'day_Fri': [day_fri], 
        'day_Sat': [day_sat],
        'day_Sun': [day_sun], 
        'time_Dinner': [time_dinner]
    })

    # Align columns with training data
    for col in X_train.columns:
        if col not in new_data:
            new_data[col] = 0
    new_data = new_data[X_train.columns]

    # Make prediction
    predicted_tips = best_model.predict(new_data)
    st.write(f"### Predicted Tip: {predicted_tips[0]:.2f}")
    st.write(f"### in above inputs 1 indicates answer 'yes' and 0 indicates 'No' ")
