# House-Price-Prediction    
"Machine Learning House price predcition using Numpy ,Pandas,Seaborn,Matplotlib,Scikit-Learn"    

--> Advanced House Price Prediction   

In this project I used multiple regression models to predict house prices using the Advanced House   Price dataset. The workflow includes data  preprocessing, feature engineering, scaling, dimensionality  reduction (PCA), model training, hyperparameter tuning, evaluation, and prediction on new data.    

 -->Project Overview    

The goal is to predict SalePrice based on various numerical and categorical features of houses.    
  
-->Project Structure   

1. Data Preprocessing (handling missing values, encoding categorical variables)    


2. Feature Engineering (creating new features like TotalSF, TotalPorchSF)    


3. Feature Scaling (using StandardScaler)      


4. Dimensionality Reduction (using PCA with 50 components)    


--> Model Training      

Linear Regression    

Ridge Regression (with GridSearchCV tuning)    

Lasso Regression (with GridSearchCV tuning)    

Random Forest Regressor (with GridSearchCV tuning)    



--> Model Evaluation using    

Mean Squared Error (MSE)    

Root Mean Squared Error (RMSE)    

R² Score    
 

 -->Steps included    

 1. Data Loading:    

Training and test datasets are loaded (Advtrain.csv & Advtest.csv).    


 2. Data Preprocessing:    

Missing values handled using SimpleImputer    

Mean imputation for numerical columns    

Most frequent imputation for categorical columns    


Label encoding for categorical columns using LabelEncoder.    


 3. Exploratory Data Analysis (EDA):    

Distribution plots of SalePrice before and after log transformation.    

Visualization saved as PNG files.    


4. Feature Engineering:      
   
-->New features:   
 I have created new features based on existing ones  

TotalSF = Total square footage (basement + 1st floor + 2nd floor)    

TotalPorchSF = Sum of all porch areas    



 5. Scaling & Dimensionality Reduction:    

StandardScaler for normalization    

PCA (50 components) to reduce dimensionality.      


 6. Model Training & Hyperparameter Tuning:        

-->Models used   

Linear Regression    

Ridge Regression (GridSearchCV tuning )    

Lasso Regression (GridSearchCV tuning )    

Random Forest Regressor (GridSearchCV tuning)    



 7. Model Evaluation Metrics:    

Mean Squared Error (MSE)    

Root Mean Squared Error (RMSE)    

R² Score      


 8. Prediction on New Data:    

A new house data sample is created, preprocessed, and predicted using the trained model.    


 9. Visualization:    

Scatter plot of Actual vs Predicted SalePrice is saved.    

-->Project Structure   

|-- Advtrain.csv    
|-- Advtest.csv    
|--house_price_prediction.py   # The main code file    
|-- Distribution of SalePrice.png    
|-- Distribution of SalePrice after log transformation.png    
|-- actual vs predicted.png    
|-- README.md  


--> Outputs  

Model evaluation metrics (MSE, RMSE, R²)    
Best hyperparameters for Ridge, Lasso, Random Forest    
SalePrice prediction for new input data    
Saved visualizations (distribution plots & prediction scatter plot)    

-->Models Used & Hyperparameter Tuning   

-->Model	Hyperparameters Tuned    

Ridge Regression	alpha, solver, max_iter    
Lasso Regression	alpha, selection, max_iter    
Random Forest	n_estimators, max_depth, max_features    

-->R^2 scores    

 Linear Regression:0.84      
 Ridge:0.85    
 Lasso:0.83    
 RandomForest:0.86    
 -->Key Features      
 Feature Engineering & PCA    
 Hyperparameter Tuning with GridSearchCV    
 Multiple Model Comparison      
 Predictions on New Data    

--> How to Run This Project   

--> Clone Repository   

git clone https://github.com/Puligorladhanush/House-Price-Prediction.git  
cd Titanic-Survival-Prediction  

 -->Install Dependencies   

pip install -r requirements.txt  

--> Run the Notebook   

jupyter notebook notebooks/Final HPP.ipynb  

 -->Contact   

 Name:Dhanush Puligorla    
 Email: dhanushpuligorla@gmail.com    
 GitHub: Puligorladhanush    
