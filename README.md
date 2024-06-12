# Diabetes-detection-based-on-Retinal-abonormalities

This project aims to detect diabetes based on retinal abnormalities using machine learning algorithms. The project includes data preprocessing, model training, and a web application for prediction.

Installation
To get started with the project, you need to install the required Python modules. You can do this using pip:

bash
pip install pandas
pip install matplotlib
Dataset
The dataset used for this project can be found on Kaggle. You can download it from the following link:

[Diabetic Retinopathy Detection Dataset](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)

Project Structure
The project consists of the following files:

Model.py: Contains the machine learning model for diabetes detection.
Partitioning.py: Handles the partitioning of the dataset into training and testing sets.
app.py: Flask application for deploying the model and making predictions.
home.html: HTML file for the web application's home page.
test_Model.py: Script for testing the machine learning model.
trainLabels.csv: CSV file containing the labels for the training data.

To run the web application, use the following command:

bash
python app.py
Then, open your web browser and go to http://localhost:5000 to access the application.

