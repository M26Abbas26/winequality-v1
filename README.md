# winequality-v1
Wine Quality Prediction API
Predictive API that assesses wine quality based on its physicochemical properties, showcasing end-to-end data analysis and machine learning deployment skills.

This project encompasses the entire data science pipeline, from exploratory data analysis to deploying a predictive model as an API. The goal is to predict wine quality based on various features like acidity, sugar content, and alcohol percentage.

**Technologies Used**
Data Analysis and Modeling: Python, Pandas, Scikit-learn
API Development: Flask
Deployment: Heroku

**Installation and Setup**
Clone the repository:

bash
git clone [Your Repository Link]
Navigate to the project directory and install the required packages:

pip install -r requirements.txt
Run the Flask API locally:
python wine_quality_api.py

**API Endpoints**

Red Wine Prediction:
Endpoint: /predict/red
Method: POST
Data Format: {"features": [feature1, feature2, ...]}

White Wine Prediction:
Endpoint: /predict/white
Method: POST
Data Format: {"features": [feature1, feature2, ...]}

**Future Enhancements**
Integration with a front-end dashboard for interactive analysis.
Incorporation of additional wine datasets to improve prediction accuracy.
Exploration of alternative machine learning models and hyperparameter tuning.
Acknowledgements
Data sourced from the UCI Machine Learning Repository.


