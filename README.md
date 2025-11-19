\# 🚗 Used Car Price Prediction App



This project is a web-based application that predicts the market value of used cars using machine learning algorithms. It features a user-friendly interface built with Streamlit, allowing users to get instant price estimates based on vehicle specifications.



!\[Python](https://img.shields.io/badge/Python-3.8%2B-blue)

!\[Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

!\[XGBoost](https://img.shields.io/badge/Model-XGBoost-green)



\## 🎯 About the Project



This application is powered by an \*\*XGBoost\*\* model trained on a dataset of thousands of vehicle sales. The project encompasses the entire data science lifecycle, including data analysis, feature engineering, and model optimization.



\### Key Features

\* \*\*High Accuracy:\*\* Utilizes an optimized XGBoost algorithm for precise predictions.

\* \*\*Interactive UI:\*\* Modern and responsive user experience designed with Streamlit.

\* \*\*Comprehensive Analysis:\*\* Considers various factors such as make, model, year, transmission, and condition.



\## 🛠️ Installation \& Usage



Follow these steps to run the project on your local machine:



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone https://github.com/Vugar-Mirzoyev/Car-Price-Prediction.git

&nbsp;   cd Car-Price-Prediction

&nbsp;   ```



2\.  \*\*Install dependencies:\*\*

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```

&nbsp;   \*(Note: The `car\_prices.csv` dataset is excluded from the repository due to size constraints. The application runs using the pre-trained `.joblib` model files.)\*



3\.  \*\*Run the application:\*\*

&nbsp;   ```bash

&nbsp;   streamlit run app.py

&nbsp;   ```



\## 📊 Tech Stack



\* \*\*Data Processing:\*\* Pandas, NumPy

\* \*\*Machine Learning:\*\* XGBoost, Scikit-learn

\* \*\*Interface:\*\* Streamlit

\* \*\*Visualization:\*\* Matplotlib, Altair



\## 📁 Project Structure



\* `app.py`: The main Streamlit application file.

\* `CARS.ipynb`: Jupyter Notebook containing data analysis, cleaning, and model training steps.

\* `\*.joblib`: Pre-trained model, scaler, and encoder files.

\* `requirements.txt`: List of project dependencies.



---

\*This project was developed as a Data Science portfolio project.\*

