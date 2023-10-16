#!/usr/bin/env python
# coding: utf-8

# In[14]:





# In[19]:


import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report





img="""
<style>
background-image:url("CR.png");
background-size=cover;

</style>
"""
Submit = """
    <style>
        .black-button button {
            background-color: black;
            color: white;
        }
    </style>

    <div class="black-button">
        <button>Black Button</button>
    </div>
"""
from streamlit_option_menu import option_menu
with st.sidebar:

   choice=option_menu (
     menu_title="Main menu",
     options=['Home','Soil predictor','Crop predictor','Help',],
     
     default_index=0,
     
     

)



d=pd.read_csv("C:\\Desktop\\hack\\CRD.csv")
le_label=LabelEncoder()
d['crop_n']=le_label.fit_transform(d['label'])
inputs=d[['N','P','K','temperature','rainfall']]
km=KMeans(n_clusters=22,random_state=0)
m=km.fit(inputs[['N','P','K','temperature','rainfall']])


df = pd.DataFrame({"N":[], "P":[],"K":[],"temp":[],"rain":[]})

                              
            
         
         


    
if choice =="Home":
    def center_text(text):
        return f"<h1 style='text-align: center;'>{text}</h1>"
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    centered_text = center_text("FARM ASSISTANCE MODEL")
    st.markdown(centered_text, unsafe_allow_html=True)
    

# Create a button that triggers the pop-up dialog



    

    

# Add CSS to set the background image
    
# Add content to your Streamlit app



   
if choice == "Soil predictor":
   
    DATADIR = "C:\\Users\\devan\Downloads\\archive (3)\\Soil types"
    CATEGORIES = ["black soil", "Cinder soil", "Laterite Soil", "Peat soil", "Yellow soil"]
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to soil types directory
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            break
        break
    
    IMG_SIZE = 225
    
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    training_data = []

    def create_training_data():
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)  # path to soil types directory
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img))
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    
                    training_data.append([new_array, class_num])
                except Exception as e:
                    pass
 
    create_training_data()

    import random
    random.shuffle(training_data)
    
    X = []
    y = []
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, 225, 225, 3)

    
    X = np.reshape(X, (X.shape[0], -1))
    Xtrain, Xtest, y_train, y_test = train_test_split(X, y, random_state=0)

    pipe = Pipeline([('SVC', SVC(kernel='rbf', C=10))])
    pipe.fit(Xtrain, y_train)

   


# Display file uploader widget
   

    


    if uploaded_file is not None:
    # Read the contents of the file as bytes
         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    
         st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
         image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
        
         image = cv2.resize(image, (225, 225))
 
         normalized_image =image / 255.0
         image1 = normalized_image.reshape(-1, 225,225,3)

         image1 = np.reshape(image, (image.shape[0], -1))

# Pass the image to the model for prediction
         predictions = pipe.predict(image1.reshape(1, -1))
         ans=predictions[0]
         if(ans==0):
             st.success("BLACK SOIL")
         if(ans==1):
             st.success("CINDER SOIL")
         if(ans==2):
             st.success("LATERITE SOIL")
         if(ans==3):
             st.success("PEAT SOIL")
         if(ans==4):
             st.success("YELLOW SOIL")        
   
if  choice=="Help":
    st.markdown(img,unsafe_allow_html=True)
    
    text =st.text_input("enter your problems")
    import re
    ctr=0
    pattern = r'\b(rain|rainfall)\b'
    pattern2 = r'\b(temperature|temp)\b'
    pattern3 = r"\b(nitrogen|nitro|soil|npk)\b"
    pattern4 = r"\b(phosphorus|phos|soil|Npk)\b"
    pattern5 = r"\b(potassium|pota|soil)\b"
    pattern6=  r'\b(list|menu|dropdown)\b'
    matches1=re.findall(pattern,text)
    matches2=re.findall(pattern2,text)
    matches3=re.findall(pattern3,text)
    matches4=re.findall(pattern4,text)
    matches5=re.findall(pattern5,text)
    matches6=re.findall(pattern6,text)

    a=0
    if len(text) > 0:
        ctr=ctr+1
    
         

    if len(matches1)==0 and len(matches2)==0 and len(matches3)==0  and len(matches4)==0 and len(matches5 )==0  and len(matches6)==0 and ctr>0 :
         st.write("sorry cannot understand the issue")
    if ctr>0 and a==0 :
         a=1
         if len(matches6)!=0:
             st.write("Sorry for the inconvinience please select an option from the list")
         if len(matches2) !=0:
              import requests

   # API endpoint and parameters
         
              st.write("pls visit the below site ")
              openweathermap_url = "https://openweathermap.org/"

              st.markdown(f"[OpenWeatherMap](https://openweathermap.org/)")
              st.markdown("<hr>", unsafe_allow_html=True)
         if len(matches1) != 0:
             rain_issue = st.radio("Choose your problem:", ["How to know the amount of rain in an area in cm", "In which unit do i have to enter the rainfall", "Can i use irrigation instead instead of rainfall","Other"],index=3)
             if rain_issue == "How to know the amount of rain in an area in cm" :
                      states = ["STATES","ANDAMAN & NICOBAR ISLANDS", "ARUNACHAL PRADESH", "ASSAM", "ORISSA", "JHARKHAND", "BIHAR",
                             "UTTAR PRADESH", "UTTARAKHAND", "HARYANA DELHI & CHANDIGARH", "PUNJAB", "HIMACHAL PRADESH",
                                "JAMMU & KASHMIR", "RAJASTHAN", "MADHYA PRADESH", "GUJARAT", "MAHARASHTRA", "CHHATTISGARH",
                                      "ANDHRA PRADESH", "TELANGANA", "TAMIL NADU", "KARNATAKA", "KERALA"]

                      selected_state = st.selectbox("Select a state", states,index=0)
                      season = st.selectbox("Select season", ["Summer", "Winter", "Moonsoon"])
                     
                      if selected_state=="ANDAMAN & NICOBAR ISLANDS":
                          
                          if season=="Summer":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Winter":
                              st.write("Average rainfall is betwenn 150cm-300 cm")
                          if season=="Moonsoon":
                              st.write("Average rainfall is between 1600-1800cm")
                      
                      elif selected_state == "ARUNACHAL PRADESH":
                             





                             if season=="Summer":
                                  st.write("Average rainfall is between 500cm-700cm")
                             if season=="Winter":
                              st.write("Average rainfall is between 50cm-150cm")
                             if season=="Moonsoon":
                              st.write("Average rainfall is between 1900cm-2100cm")
                      elif selected_state == "ASSAM":
                             
                             if season=="Summer":
                               st.write("Average rainfall is between 400cm-600cm")
                             if season=="Winter":
                              st.write("Average rainfall is between 400cm-600cm")
                             if season=="Moonsoon":
                               st.write("Average rainfall is between 400cm-600cm")
                      elif selected_state == "ORISSA":
                        
                          if season=="Summer":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Winter":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Moonsoon":
                              st.write("Average rainfall is between 400cm-600cm")
                      elif selected_state == "JHARKHAND":
                          if season=="Summer":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Winter":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Moonsoon":
                              st.write("Average rainfall is between 400cm-600cm")
                          
                      elif selected_state == "BIHAR":
                          if season=="Summer":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Winter":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Moonsoon":
                              st.write("Average rainfall is between 400cm-600cm") 
                         
                      elif selected_state == "UTTAR PRADESH":
                          if season=="Summer":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Winter":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Moonsoon":
                              st.write("Average rainfall is between 400cm-600cm")
                        
                      elif selected_state == "UTTARAKHAND":
                          if season=="Summer":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Winter":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Moonsoon":
                              st.write("Average rainfall is between 400cm-600cm")
                          
                      elif selected_state == "HARYANA DELHI & CHANDIGARH":
                          if season=="Summer":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Winter":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Moonsoon":
                              st.write("Average rainfall is between 400cm-600cm")
                        
                      elif selected_state == "PUNJAB":
                          if season=="Summer":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Winter":
                              st.write("Average rainfall is between 400cm-600cm")
                          if season=="Moonsoon":
                              st.write("Average rainfall is between 400cm-600cm")
                         
                      elif selected_state == "HIMACHAL PRADESH":
                           if season=="Summer":
                              st.write("1")
                           if season=="Winter":
                              st.write("200-300")
                           if season=="Moonsoon":
                              st.write("1")
                           
                      elif selected_state == "JAMMU & KASHMIR":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                         
                      elif selected_state == "RAJASTHAN":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                       
                      elif selected_state == "MADHYA PRADESH":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                           
                      elif selected_state == "GUJARAT":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                      elif selected_state == "MAHARASHTRA":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("0")
                      elif selected_state == "CHHATTISGARH":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                      elif selected_state == "ANDHRA PRADESH":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                      elif selected_state == "TELANGANA":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                      elif selected_state == "TAMIL NADU":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                      elif selected_state == "KARNATAKA":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                      elif selected_state == "KERALA":
                          if season=="Summer":
                              st.write("1")
                          if season=="Winter":
                              st.write("200-300")
                          if season=="Moonsoon":
                              st.write("1")
                  
                     
             if rain_issue =="In which unit do i have to enter the rainfall":
                 st.write("You have to enter rainfall in cm")
                 st.markdown("<hr>", unsafe_allow_html=True)
             if rain_issue =="Can i use irrigation instead instead of rainfall":
                 
                 st.write("Yes u can use irrigation to compensate for lack of rainfall")
                 st.markdown("<hr>", unsafe_allow_html=True)
         if len (matches3) or len(matches4) or len(matches5) != 0:
             
             st.write("please see the video for information regarding how to find npk values of your soil ")
             
             video_url = "https://www.youtube.com/watch?v=iE9P9w62l2Y"

             st.markdown(f"[Click here to watch the video]({video_url})")
             st.markdown("<hr>", unsafe_allow_html=True)
              
                      
               

                     
                 

                 
    
if choice=='Crop predictor':
    st.title("Crop prediction model")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""   ## Enter details""")
    nitro =st.text_input("enter nitrogen content( mg/kg)")
    phos=st.text_input("enter phosphorous ( mg/kg )")
    pota =st.text_input("enter your potasium( mg/kg)")
    temp=st.text_input("enter temperature in celcius")
    rain=st.text_input("enter rainfall in cm")
    arr=[]
    
    if st.button("Submit"):
        df=df.append({"N":nitro,"P":phos,"K":pota,"temp":temp,"rain":rain}, ignore_index=True)
        df.to_excel("C:\\Desktop\\hack\\back.xlsx", index = False)
        st.write("records saved ")
        
        if nitro=="" or pota =="" or phos =="" or temp =="" or rain =="":
            st.write("enter all values")
        
        nitro1=int(nitro) 
        phos1=int(phos)
        pota1=int(pota)
        temp1=int(temp)
        rain1=int(rain)
        arr=m.predict([[nitro1,phos1,pota1,temp1,rain1]])   
        def closePopup():
           st.write("Pop-up window closed.")
           st.session_state["close_button_clicked"] = False
        if arr[0] == 0:
            st.success("You can grow kidneybeans, pigeonpeas, blackgram, and mothbean")
        elif arr[0] == 1:
            st.success("You can grow coffee, jute, and papaya")
        elif arr[0] == 2:
            st.success("You can grow grapes")
        elif arr[0] == 3:
            st.success("You can grow maize and papaya")
        elif arr[0] == 4:
            st.success("You can grow papaya, mango, and promogranate")
        elif arr[0] == 5:
           
           

             st.success("You can grow rice ")
        elif arr[0] == 6:
            st.success("You can grow chickpea and papaya")
        elif arr[0] == 7:
            st.success("You can grow watermelon")
        elif arr[0] == 8:
            st.success("You can grow maize and cotton")
        elif arr[0] == 9:
            st.success("You can grow rice and papaya")
        elif arr[0] == 10:
            st.success("You can grow pigeonpeas and papaya")
        elif arr[0] == 11:
            st.success("You can grow rice, coffee, jute, and papaya")
        elif arr[0] == 12:
            st.success("You can grow banana")
        elif arr[0] == 13:
            st.success("You can grow muskmelon")
        elif arr[0] == 14:
            st.success("You can grow mugbeans and mothbeans")
        elif arr[0] == 15:
            st.success("You can grow coconut")
        elif arr[0] == 16:
            st.success("You can grow apple")
        elif arr[0] == 17:
            st.success("You can grow coconut")
        elif arr[0] == 18:
            st.success("You can grow lentill, mugbeans, and mothbean")
        elif arr[0] == 19:
            st.success("You can grow orange")
        elif arr[0] == 20:
            st.success("You can grow maize, lentill, blackgram, mugbeans, and mothbeans")
        elif arr[0] == 21:
            st.success("You can grow kidneybeans, pigeonpeas, and papaya")
    





