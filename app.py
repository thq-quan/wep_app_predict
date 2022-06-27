# khai báo thư viện cần dùng
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import category_encoders as ce
# load data
data = pd.read_csv("D:\streamlit\data_final_clean.csv")
X = data.drop(['is_paid'], axis=1)
y = data['is_paid']
## biến đổi kiểu dữ liệu 
encoder = ce.OrdinalEncoder(cols=['region','language','package'])
X_model = encoder.fit_transform(X)
#
id3 = DecisionTreeClassifier(criterion="entropy")
id3.fit(X_model,y)
# title
st.write("""
# Prediction App for user iRender
This app predicts the **User paid**!
""")
# data của ng dùng nhập vào để dự đoán
st.sidebar.header('User Input Parameters')
re = data['region']
re_list = list(set(re))
la = data['language']
la_list = list(set(la))
pa = data['package']
pa_list = list(set(pa))

def user_input_features():
    region = st.sidebar.selectbox('Region',re_list)
    timezone = st.sidebar.slider('Timezone',-12,12,0)
    language = st.sidebar.selectbox('Language',la_list,)
    lasttime = st.sidebar.text_input('Last time active (day)','0'),
    created_bill = st.sidebar.text_input('Create acc to first bill(hours)','0')
    package = st.sidebar.selectbox('Package',pa_list)
    hours_use = st.sidebar.text_input('hours_use(hours)','0')
    sum_length = st.sidebar.text_input('sum_length(GB)','0')
    data = {'region': region,
            'timezone': timezone,
            'language': language,
            'lasttime': lasttime,
            'created_bill': created_bill,
            'package': package,
            'hours_use': hours_use,
            'sum_length': sum_length
            }
    features = pd.DataFrame(data, index=[0])
    return features
# kq sau khi ng dùng nhập   
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)
# biến đổi dữ mà ng dùng nhập
df = encoder.transform(df)
# hàm dự đoán và tỷ lệ giữa 2 lớp
prediction = id3.predict(df)
prediction_proba = id3.predict_proba(df)
# show dự đoán của mô hình
st.subheader('Prediction')
paid = np.array(['free','paid'])
st.write(paid[prediction])
# show tỉ lệ giữa 2 lớp
st.subheader('Prediction Probability')
st.write(prediction_proba)