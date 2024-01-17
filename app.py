import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import f1_score,accuracy_score,precision_score ,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle 
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

page=st.sidebar.selectbox("Выберите страницу",['Датасет','Предикт'])
if page=='Датасет':
  "A_id: Unique identifier for each fruit"
  "Size: Size of the fruit"
  "Weight: Weight of the fruit"
  "Sweetness: Degree of sweetness of the fruit"
  "Crunchiness: Texture indicating the crunchiness of the fruit"
  "Juiciness: Level of juiciness of the fruit"
  "Ripeness: Stage of ripeness of the fruit"
  "Acidity: Acidity level of the fruit"
  "Quality: Overall quality of the fruit"
elif page=='Предикт':
  st.title("Предикт модели по введеным данным")
  st.subheader("Введите данные для предсказания:")
  input_data={}
  feature_names=['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness',
       'Acidity']
  for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", min_value=0.0, value=10.0)
  if st.button('Сделать предсказание'):
        xgb_model_loaded = pickle.load(open("xgboost.pkl", "rb"))
        light_loaded=pickle.load(open("light.pkl","rb"))
        dt_loaded=pickle.load(open("dt.pkl","rb"))
        input_df = pd.DataFrame([input_data])
        st.write("Входные данные:", input_df)

        # Сделать предсказания на тестовых данных
        predictions_ml1 = xgb_model_loaded.predict(input_df)
        predictions_ml2=light_loaded.predict(input_df)
        predictions_ml3=dt_loaded.predict(input_df) 
        st.success(f"Предсказанние XGboost Classifier: {predictions_ml1}")
        st.success(f"Предсказание Light:{predictions_ml2}")
        st.success(f"Предсказание DT:{predictions_ml3}")
