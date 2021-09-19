import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import requests
import streamlit as st

@st.cache(allow_output_mutation=True)
model = tf.keras.models.load_model('moneyheist_checkpoint.h5')
def scale(image):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return tf.image.resize(image,[224,224])

def decode_img(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = scale(img)
    return np.expand_dims(img,axis=0)

classes = ['One thousand vnd', 'Two thousand vnd','Five thousand vnd', 'Ten thousand vnd', 'Twwenty thousand vnd', 'Fifty thousand vnd', 'A hundred thousand vnd', 'twwo hundred thousand vnd', 'Five hundred thousand vnd']

menu = ['Home', 'Upload Image', 'Image from Internet', 'About Me']
choice = st.sidebar.selectbox('Vietnamese cash classifier',menu)
if choice=='Home':
    st.title("Money heist for beginner: A classifier")
    st.header("Web-based cash classifier wwith accurcay of 96.45%")

    st.write("")
    st.write("Wanna start a heist but not familiar with Vietnamese cash?")
    st.write("You will find this tool useful")

    st.image('di duong quyen.gif',
              use_column_width='auto')

elif choice=='Upload Image':
    st.title('Upload your cash image, totally safe (cuz I am too lazy to steal)')
    photo_uploaded = st.file_uploader('Upload your best photo here', ['png', 'jpeg', 'jpg'])
    if photo_uploaded!=None:
        img = decode_img(photo_uploaded.read())
        st.image(img, channels='BGR')
        st.write('Predicted class:')
        with st.spinner('Classifying...'):
            label = np.argmax(model.predict(img),axis=1)
            st.write(classes[label[0]])
            st.write('')

elif choice=='Image from Internet':
    path = st.text_input('Enter Image Url to classify', 'https://upload.wikimedia.org/wikipedia/vi/9/9f/500000_polymer.jpg')
    if path is not None:
        content = requests.get(path).content
        st.write('Predicted class:')
    with st.spinner('Classifying...'):
        img=decode_img(content)
        st.image(img, channels='BGR')
        label = np.argmax(model.predict(img),axis=1)
        st.write(classes[label[0]])

elif choice=='About Me':
    st.success('Super cute geek as you might wonder')
    st.image('4BON.gif')
    st.balloons()
