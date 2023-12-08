import streamlit as st 
import pickle as pkl
from PIL import Image
import numpy as np

class_list = {'0': 'FeMale', '1': 'Male'}
st.title('HandWritten Degit Recognition')

#image = Image.open('vi-names.png')
#st.image(image)

input_md = open('lrc_mnist.pkl', 'rb')
model = pkl.load(input_md)

st.header('Upload HandWritten Degit Image')
image = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])

if image is not None:
  image = Image.open(image)
  st.image(image, caption='Test image')
  if st.button('Predict'):
    image = image.resize((8*8, 1))
    vector = np.array(image)
    lable = str(model.predict(model.predict(vector))[0])

    st.header('Result')
    st.text(class_list[label])
    
