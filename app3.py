
# coding: utf-8

# In[6]:


import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# In[7]:


def app():
    st.title("Coronavirus 2019")
    st.write("Source : https://en.wikipedia.org/wiki/Coronavirus_disease_2019")
    st.write("Coronavirus disease 2019 (COVID-19) is a contagious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The first case was identified in Wuhan, China, in December 2019.The disease has since spread worldwide, leading to an ongoing pandemic")
    st.write("The virus that causes COVID-19 spreads mainly when an infected person is in close contact with another person. Small droplets and aerosols containing the virus can spread from an infected person's nose and mouth as they breathe, cough, sneeze, sing, or speak. Other people are infected if the virus gets into their mouth, nose or eyes. The virus may also spread via contaminated surfaces, although this is not thought to be the main route of transmission. The exact route of transmission is rarely proven conclusively, but infection mainly happens when people are near each other for long enough. People who are infected can transmit the virus to another person up to two days before they themselves show symptoms, as can people who do not experience symptoms. People remain infectious for up to ten days after the onset of symptoms in moderate cases and up to 20 days in severe cases. Several testing methods have been developed to diagnose the disease. The standard diagnostic method is by detection of the virus' nucleic acid by real-time reverse transcription polymerase chain reaction (rRT-PCR), transcription-mediated amplification (TMA), or by loop-mediated isothermal amplification from a nasopharyngeal swab.")
    st.subheader("Symptoms of covid-19")
    img=Image.open("Images/symptoms.png")
    newsize=(600,600)
    img1=img.resize(newsize)
    st.image(img1)

