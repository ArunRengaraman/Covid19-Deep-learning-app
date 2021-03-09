
# coding: utf-8

# In[12]:


import streamlit as st
import numpy as np
import keras
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# In[13]:


class_dict = {0:'COVID19',
              1:'NORMAL',
              2:'PNEUMONIA'}

model_list={"VGG-16":'vgg16',"VGG-19":'vgg19',"DenseNet":'densenet',"Resnet50":"resnet50","Inception":"inception","NasNet":'nasnet',"EfficientNet":'efficientnet'}

model_rootpath='models/'

image_rootpath='Images/'
def app():
    st.title("Covid-19 prediction using different Deep learning pretrained models")
    st.subheader("           Play with different models by selecting them from the sidebar")
    st.write("NOTE : Our Dataset performs well only on VGG16 model")
    clf_model=st.sidebar.radio("Choose the model",("VGG-16","VGG-19","DenseNet","Resnet50","Inception","NasNet","EfficientNet"))
    
    if clf_model=='VGG-16':
        st.write("VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”")
        st.write("The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous model submitted to ILSVRC-2014")
        st.write("It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.")
        st.subheader("VGG-16 Model architecture")
        st.image(image_rootpath+'vgg16_arch.png')
        st.write("Source : https://neurohive.io/en/popular-networks/vgg16/")
        st.subheader("VGG 16 performance visualization")
        st.image(image_rootpath+'vgg16_runtime.jpg')
        st.image(image_rootpath+'vgg16_train_vs_test.png')
        st.image(image_rootpath+'vgg16_train_vs_test_loss.png')
        
    elif clf_model=='VGG-19':
        st.write("VGG-19 is a convolutional neural network that is 19 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database [1]. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224")
        st.write("Source : https://www.mathworks.com/help/deeplearning/ref/vgg19.html")
        st.subheader("VGG-19 Model architecture")
        st.image(image_rootpath+'vgg19_arch.jpg')
        st.subheader("VGG 19 performance visualization")
        st.image(image_rootpath+'vgg19_runtime.jpg')
        st.image(image_rootpath+'vgg19_train_vs_test.png')
        st.image(image_rootpath+'vgg19_train_vs_test_loss.png')
        
    elif clf_model=='DenseNet':
        st.write("Introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections — one between each layer and its subsequent layer — our network has L(L+1)/ 2 direct connections")
        st.write("For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers.")
        st.write("DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.")
        st.write("Source : https://towardsdatascience.com/densenet-2810936aeebb")
        st.subheader("DenseNet model architecture")
        st.image(image_rootpath+"densenet_arch.png")
        st.subheader("DenseNet performance visualization")
        st.image(image_rootpath+'densenet_runtime.jpg')
        st.image(image_rootpath+'densenet_train_vs_test.png')
        st.image(image_rootpath+'densenet_train_vs_test_loss.png')
    
    elif clf_model=='Resnet50':
        st.write('ResNet50 is a variant of ResNet model which has 48 Convolution layers along with 1 MaxPool and 1 Average Pool layer. It has 3.8 x 10^9 Floating points operations. It is a widely used ResNet model and we have explored ResNet50 architecture in depth.')
        st.write("Source : https://iq.opengenus.org/resnet50-architecture/")
        st.subheader("ResNet50 Model architecture")
        st.image(image_rootpath+"resnet50_arch.png")
        st.subheader("ResNet performance visualization")
        st.image(image_rootpath+'resnet50_runtime.png')
        st.image(image_rootpath+'resnet50_train_vs_test.png')
        st.image(image_rootpath+'resnet_train_vs_test_loss.png')
    
    elif clf_model=='Inception':
        st.write("Inception is a deep convolutional neural network architecture that was introduced in 2014. It was mostly developed by Google researchers. Inception’s name was given after the eponym movie.")
        st.subheader("Inception Model architecture")
        st.image(image_rootpath+'inception_arch.png')
        st.write("Source : https://arxiv.org/pdf/1409.4842v1.pdf")
        st.subheader('Inception performance visualization')
        st.image(image_rootpath+'inception_runtime.png')
        st.image(image_rootpath+'inception_train_vs_test.png')
        st.image(image_rootpath+'inception_train_vs_test_loss.png')
        
    elif clf_model=='NasNet':
        st.write("NASNet-Large is a convolutional neural network that is trained on more than a million images from the ImageNet database . The network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images.")
        st.write("Source : https://in.mathworks.com/help/deeplearning/ref/nasnetlarge.html")
        st.subheader("NasNet model architecture")
        st.image(image_rootpath+'nasnet_arch.jpeg')
        st.subheader("NasNet performance visualization")
        st.image(image_rootpath+'nasnet_runtime.jpeg')
        st.image(image_rootpath+'nasnet_train_vs_test.png')
        st.image(image_rootpath+'nasnet_train_vs_test_loss.png')
    
    else:
        st.write("EfficientNet model was proposed by Mingxing Tan and Quoc V. Le of Google Research, Brain team in their research paper ‘EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks’. This paper was presented in the International Conference on Machine Learning, 2019. These researchers studied the model scaling and identified that carefully balancing the depth, width, and resolution of the network can lead to better performance.")
        st.write("Based on this observation, they proposed a new scaling method that uniformly scales all dimensions of depth, width and resolution of the network. They used the neural architecture search to design a new baseline network and scaled it up to obtain a family of deep learning models, called EfficientNets, which achieve much better accuracy and efficiency as compared to the previous Convolutional Neural Networks.")
        st.write("Source : https://analyticsindiamag.com/implementing-efficientnet-a-powerful-convolutional-neural-network/")
        st.subheader("EfficientNet model architecture")
        st.image(image_rootpath+'efficientnet_arch.jpeg')
        st.subheader('Efficient performance visualization')
        st.image(image_rootpath+'efficientnet_runtime.png')
        st.image(image_rootpath+'efficientnet_train_vs_test.png')
        st.image(image_rootpath+'efficientnet_train_vs_test_loss.png')
        
    clf=load_model(model_rootpath+model_list[clf_model]+'.h5')   
    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    
    if st.button("Process"):
        test_image = Image.open(image_file)
        test_image = np.array(test_image.convert('RGB'))
        test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)
        st.image(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        probs = clf.predict(test_image)
        pred_class = np.argmax(probs)

        pred_class = class_dict[pred_class]

        st.write('prediction: ',pred_class)
        

