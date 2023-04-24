# [Online data] : Titanic DS : Data Science Dashboard in  Streamlit  

import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
# we need to make four containers now : 

header = st.container();
data_sets = st.container();
features = st.container();
model_training = st.container();

# starting with header now :
with header:
    st.title("Titanic App ")
    # header : 
    st.text("In this project we will be working with the titanic dataset  ")

with data_sets:
    st.header("Titanic is sanked in !")
    st.text("we wil lwrok with titanic data sets  ")
    # importing data here :
    df = sns.load_dataset('titanic')
    # we need to drop na from age ds 
    df = df.dropna()
    st.write(df.head(10))
    # now we have loaded the ds 
    st.subheader("How many people were there ? ")
    st.bar_chart(df['sex'].value_counts())
    # value_counts()) -- > made all the Garbage representation disappear : 
    
    # to draw other plots :
    st.subheader("Graphical  Representation wrt to class : ")
    st.bar_chart(df['class'].value_counts())

    # now to give this 2 graphs labels we use :

    # another plots 
    st.subheader("Graphical  Representation wrt to sex : ")
    st.bar_chart(df['age'].sample(10))



with features:
    st.header("These are our features :  ")
    st.text("we have so many features here !   ")
    # now starting with feeatures part :

    st.markdown('1. **Feature 1:**  This will tell us ')
    st.markdown('1. **Feature 2:**  This will tell us ')
    st.markdown('1. **Feature 3:**  This will tell us ')

# now container & features are done , how will we train the model now ? 
# we need to make 2 columns 
with model_training:
    st.header("Titanic - Model  Training ")
    st.text("we will train the model !   ")

# we have created the four containers now ! 

# now wee need to download datasets 
# import seadborn and pandas now 

# now dataset ke container mai import data karna hai 
# if online --> df 


    # making columns 

    input, display = st.columns(2)

    # first col : we need to have selection point 
    max_depth = input.slider(" how many people do you know ? " , min_value=10 , max_value= 100 , value = 20 , step=1 )
    
    # n_estimators designing now : 
    n_estimators = st.selectbox("How many trees should be there in a random forest ", options=[50,100,200,300,'NO limit'])

# adding the list of features :
input.write(df.columns)

    # another column 
inputs_features = input.text_input('which featurs should we use ')

    # overall so many things are done now : 
    # now we need to apply the ml algorithm 

#    from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score

# to make a ml model  :
model = RandomForestRegressor(max_depth=max_depth , n_estimators=n_estimators)

# we need to fit the model : define x and y

x = df[[inputs_features]]
y = df[['fare']]

# fit our model 
model.fit(x,y)
pred = model.predict(y)

# where is the model actually runnign : display matrices 

# 2 col initailly tha one side all these above and next side col mai displaying : 

display.subheader("Mean absolute error of the model is : ")
display.write(mean_absolute_error(y, pred))

display.subheader("Mean squared error of the model is : ")
display.write(mean_squared_error(y, pred))

display.subheader("R squared error of the model is : ")
display.write(r2_score(y, pred))

# add all the features : 