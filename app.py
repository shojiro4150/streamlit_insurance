# 基本ライブラリー
import numpy as np
import pandas as pd
import streamlit as st
# データセット
## データの読み込み
train = pd.read_csv("insurance.csv")

##Converting objects labels into categorical
train[['sex', 'smoker', 'region']] = train[['sex', 'smoker', 'region']].astype('category')

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(train.sex.drop_duplicates())
train.sex = label.transform(train.sex)
label.fit(train.smoker.drop_duplicates())
train.smoker = label.transform(train.smoker)
label.fit(train.region.drop_duplicates())
train.region = label.transform(train.region)

## 学習済みモデルの復元

import pickle
with open('lgb_best.pickle', mode='rb') as fp:
    lgb_best = pickle.load(fp)


# アプリ
import streamlit as st
## サイドパネル（インプット部）
st.sidebar.header('Input Features')

### age入力（スライドバー）
minValue_age = int(np.floor(train['age'].min()))
maxValue_age = int(np.ceil(train['age'].max()))
startValue_age =int((maxValue_age+minValue_age)/2)
ageValue = st.sidebar.slider('age', min_value=minValue_age, max_value=maxValue_age, step=1, value=startValue_age)

### sex入力（ラジオボタン）
sexValue = st.sidebar.radio('sex',("male", "female"))

### bmi入力（スライドバー）
minValue_bmi = int(np.floor(train['bmi'].min()))
maxValue_bmi = int(np.ceil(train['bmi'].max()))
startValue_bmi =int((maxValue_bmi+minValue_bmi)/2)
bmiValue = st.sidebar.slider('bmi', min_value=minValue_bmi, max_value=maxValue_bmi, step=1, value=startValue_bmi)

### children入力（スライドバー）
minValue_children = int(np.floor(train['children'].min()))
maxValue_children = int(np.ceil(train['children'].max()))
startValue_children =int((maxValue_children+minValue_children)/2)
childrenValue = st.sidebar.slider('children', min_value=minValue_children, max_value=maxValue_children, step=1, value=startValue_children)

### smoker入力（ラジオボタン）
smokerValue = st.sidebar.radio('smoker',("yes", "no"))

## region入力（ラジオボタン）
regionValue = st.sidebar.radio('region',('southwest', 'southeast', 'northwest', 'northeast'))

## メインパネル（アウトプット部）
st.write("""
    ### 保険料試算アプリ (機械学習)
""")

### インプットデータ（1行のデータフレーム生成）
value_df = pd.DataFrame([],columns=['age','sex', 'bmi','children','smoker','region'])
record = pd.Series([ageValue, sexValue, bmiValue, childrenValue, smokerValue, regionValue], index=value_df.columns)
value_df = value_df.append(record, ignore_index=True)

## メインパネル（アウトプット部）
st.write("""
    #### 入力内容
""")
st.table(value_df)

#label = LabelEncoder()
#label.fit(value_df.sex.drop_duplicates())
#svalue_df.sex = label.transform(value_df.sex)
#label.fit(value_df.smoker.drop_duplicates())
#value_df.smoker = label.transform(value_df.smoker)
#label.fit(value_df.region.drop_duplicates())
#value_df.region = label.transform(value_df.region)

if value_df.sex[0] == "male":
    value_df.sex = 0
else:
    value_df.sex = 1

if value_df.smoker[0] == "yes":
    value_df.smoker = 1
else:
    value_df.smoker = 0

if value_df.region[0] == "southwest":
    value_df.region = 0
elif value_df.region[0] == "southeast":
    value_df.region = 1
elif value_df.region[0] == "northwest":
    value_df.region = 2
else:
    value_df.region = 3

value_df.age = value_df.age.astype(int)
value_df.bmi = value_df.bmi.astype(int)
value_df.children = value_df.children.astype(int)

value_df.sex = value_df.sex.astype(int)
value_df.smoker = value_df.smoker.astype(int)
value_df.region = value_df.region.astype(int)


### 予測
pred_probs = lgb_best.predict(value_df)
pred = pd.DataFrame(pred_probs).set_axis(['保険料'], axis='columns')

### 結果出力
## メインパネル（アウトプット部）
st.write("""
    #### 保険料 予測結果
""")

st.table(pred)


## メインパネル（アウトプット部）
st.write("""
    ###### 利用データ : kaggle - Medical Insurance Cost with Linear Regression
""")
## メインパネル（アウトプット部）
st.write("""
https://www.kaggle.com/code/mariapushkareva/medical-insurance-cost-with-linear-regression
""")