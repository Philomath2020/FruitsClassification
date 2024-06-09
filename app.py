import streamlit as st
from fastai.vision.all import *

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.PureWindowsPath

#titLe
st.title('Mevalarni klassifikatsiya qiluvchi model')

#malumot
st.subheader("Uzum, Qulupnay, Banan, Ananasni aniqlovchi dastur")

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg', 'jpg'])

if file:
    st.image(file)

    img = PILImage.create(file)

    #model
    model = load_learner('fruits.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')
