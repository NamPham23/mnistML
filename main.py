import streamlit as st 
import pickle as pkl
from PIL import Image
import numpy as np

class_list = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'}
st.title('HandWritten Degit Recognition')

#image = Image.open('vi-names.png')
#st.image(image)

input = open('lrc_mnist.pkl', 'rb')
model = pkl.load(input)

st.header('Upload HandWritten Degit Image')
uploaded_file = st.file_uploader('Choose an image', type=(['png', 'jpg', 'jpeg']))

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Test image')
  
  if st.button('Predict'):
    image = image.resize((8*8,1))
    #vector = np.array(image.convert('L')).reshape(1, -1)
    feature_vector = np.array(image)
    label = str((model.predict(feature_vector))[0])

    st.header('Result')
    st.text([label])
    
