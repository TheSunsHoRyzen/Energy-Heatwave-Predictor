# ⚡ Electricity Consumption Predictor

## Purpose  
The **Electricity Consumption Predictor** is a machine learning application designed to forecast electricity usage in London based on identified heatwaves and temporal factors.  
By providing accurate predictions, the tool helps users and organizations optimize energy planning, reduce costs, and promote sustainable usage.

---

##  Models & Approach  
- **Random Forest Regression (Scikit-learn)**  
  Used for robust and accurate forecasting of electricity consumption, leveraging its ability to capture nonlinear relationships and to handle high-dimensional data. 
  We achieved an r^2 score of .94 after parameter tuning.
- **Additional ML Models **  
  Other regression algorithms from `scikit-learn` were explored to compare performance and validate the effectiveness of Random Forest were Linear Regression and Decision Tree Regression. 

---

##  Technologies Used  
- **Python** — core programming language for data analysis and modeling  
- **Scikit-learn** — for model training, evaluation, and feature engineering  
- **Random Forest Regression** — primary predictive model  
- **Streamlit** — deployed as an interactive web application for real-time predictions  
- **Pandas & NumPy** — data manipulation and preprocessing  

---

##  Deployment  
The application is deployed using **Streamlit**, providing a simple and user-friendly interface to input parameters and view electricity consumption forecasts instantly.

---

##  Demo 


https://github.com/user-attachments/assets/69925e8e-9361-42bd-b6ac-360feef6d276



---

##  Future Improvements  
- Incorporate additional models such as Gradient Boosting and Neural Networks for comparison  
- Extend data sources to include a wider breadth of London temperature data
- Allow for user adjustable heatwave temperature threshold

---
