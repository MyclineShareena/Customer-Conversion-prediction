-->> Customer Conversion Prediction using XGBoost Classifier <<--

This is a Machine Learning model developed using XGBoost Classifier that predicts whether a customer will convert or not for an insurance company. 
The model is deployed in a Streamlit web app for easy usage in the real world usecases.

-->> Data <<--

The dataset used for this project contains information about the customers, such as their age, gender, salary, etc. 
It also contains information about their interactions with the company, such as the number of calls they made to customer service, the type of policy they are interested 
in, etc. The dataset contains both categorical and numerical variables.

-->> Model <<--

The XGBoost Classifier was chosen for this project due to its ability to handle both categorical and numerical variables, as well as its high accuracy and efficiency. 
The model was trained on a portion of the dataset using cross-validation to optimize hyperparameters, and then tested on a holdout set to evaluate its performance.

The final model achieved an accuracy of 85% and an AUC score of 0.92, indicating that it is a reliable predictor of customer conversion.

-->> Deployment <<--

The model is deployed in a Streamlit web app for easy interaction. The app allows the user to input information about a potential customer and 
receive a prediction of whether they are likely to convert or not. The app also includes visualizations of the data and model performance metrics.

To run the app locally, follow these steps:

1 .Clone the repository to your local machine.

2 .Install the required packages by running pip install -r requirements.txt in the project directory.

3 .Run "streamlit run deploy.py" to start the app.

4 .Access the app by the link created by the ngrok to tunnel to the local host.

-->> Usage <<--

To use the app, simply fill out the input form with information about a potential customer, such as their age, gender, education, marital status, etc. 
The app will then make a prediction of whether they are likely to convert or not based on the trained XGBoost Classifier model. 

-->> Credits <<--

This project was developed by Tamizharasan Govindasamy . The XGBoost Classifier model was built using the XGBoost library, and the web app was built 
using Streamlit. The dataset used for this project was provided by Guvi for the educational purpose.


Some of the images of the web app are attached below. To get more understanding please do check out the code,

![image](https://user-images.githubusercontent.com/119114780/221541799-b497bf7d-2a70-43d1-9252-ed2790b512b1.png)
![image](https://user-images.githubusercontent.com/119114780/221541814-b6df7c4a-8e54-49af-8825-55f7e1c24867.png)
->> Response by the app for a customer details.
![image](https://user-images.githubusercontent.com/119114780/221541832-f7f56ac2-14d2-4796-bb15-60f57bdf8ae8.png)
->> Response by the app for a non customer details.
![image](https://user-images.githubusercontent.com/119114780/221541846-4acc5a4e-61a5-4b63-9e22-00c9eb0387e0.png)


