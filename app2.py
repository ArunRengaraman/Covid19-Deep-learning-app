
# coding: utf-8

# In[3]:


import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# In[7]:


def app():
    st.title("About us")
    img=Image.open("Images/Arun.jpg")
    newsize=(280,300)
    img1=img.resize(newsize)
    st.subheader("1.Arun R")
    st.image(img1)
    st.write("A person with curiosity and enthusiasm in knowing about hardware components of a PC and makes surrounding filled with happiness")
    st.write("Date of birth : 21-09-1999")
    st.write("Graduation from : SASTRA DEEMED TO BE UNIVERSITY")
    st.write("Year of study : 4th year")
    st.write("Department : Computer Science and Engineering")
    st.write("Area of Interest : Machine Learning and Deep Learning")
    st.subheader("Top goals from my bucketlist")
    st.write("1.Building a gaming PC")
    st.write("2.Long trip with friends")
    st.write("3.Contriute money to orphanage")
    st.write("4.Go for a world tour")
    st.write("5.Work in game development community")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    img2=Image.open("Images/Myself.png")
    img2=img2.resize(newsize)
    st.subheader("2.Arudhra Narasimhan V")
    st.image(img2)
    st.write("A curious person who has passion to learn new software techs and not a demoraliser.")
    st.write("Date of birth : 26-06-1999")
    st.write("Graduation from : SASTRA DEEMED TO BE UNIVERSITY")
    st.write("Year of study : 4th year")
    st.write("Department : Computer Science and Engineering")
    st.write("Area of Interest : Machine Learning , Deep Learning , Graph Theory and System Designing")
    st.subheader("Top goals from my bucketlist")
    st.write("1.Go for a world tour")
    st.write("2.Have a camp fire with friends")
    st.write("3.Be a philanthrophist")
    st.write("4.Learn Organic farming")
    st.write("5.Be updated with latest software techs")

