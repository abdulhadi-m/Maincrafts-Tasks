# Task 2: Feature Engineering, Model Optimization & Performance Comparison

## Project Overview
This project extends the basic linear housing price prediction model by incorporating feature engineering, standard normalization protocols, and multi-model benchmarking. Using the **California Housing Dataset**, the goal is to evaluate, optimize, and compare different regression algorithms to determine the most effective approach for real-world continuous data forecasting.

## Key Features Implemented
* **Exploratory Data Analysis (EDA):** Structural assessment of columns, descriptive statistical matrices, and mapping linear dependencies using a Seaborn correlation heatmap.
* **Feature Engineering & Scaling:** Applied `StandardScaler` to normalize input parameters, preventing feature scale disparities from biasing model coefficients.
* **Multi-Model Pipeline:** Trained and benchmarked three distinct structural algorithms:
  1. Linear Regression (Baseline Model)
  2. Ridge Regression (L2 Regularized Linear Model)
  3. Decision Tree Regressor (Non-linear Model)
* **Model Evaluation & Diagnostics:** Evaluated variance errors using Root Mean Squared Error (RMSE) and $R^2$ scores. Generated residual analysis plots and actual-vs-predicted distribution maps.
* **Model Serialization:** Finalized model configurations exported via the `pickle` framework (`best_model.pkl`).

## Repository Folder Structure
* `Task-2_House_Price_Prediction.ipynb` : The main execution script containing the engineering pipeline and visual output graphs.
* `Task-2_Documentation.docx` : Formal comprehensive written technical report detailing metrics and theoretical formulas.
* `best_model.pkl` : Extracted serialized model weights file.
* `README.md` : Brief directory description.

## Technical Frameworks & Libraries
* **Language:** Python
* **Environment:** Jupyter Notebook
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
