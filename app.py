
# coding: utf-8

# In[47]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.models import load_model
import app1
import app2
import app3


# In[48]:


PAGES={"Deep learning models":app1,"Covid-19 History":app3,"About Us":app2}


# In[49]:


st.sidebar.title("Choose your option to navigate")
selection=st.sidebar.radio("Go to",list(PAGES.keys()))
page=PAGES[selection]
page.app()

