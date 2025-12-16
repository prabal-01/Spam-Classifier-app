import streamlit as st
import pickle
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

from nltk.tokenize import word_tokenize

tfidf=pickle.load(open('vectorize.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
st.title("SMS/Email-Spam Classifier")
input_sms=st.text_input("Enter your message")

if st.button("Predict"):

    transformed_text=transform_text(input_sms)
    vector_input=tfidf.transform([transformed_text])

    model_output=model.predict(vector_input)[0]
    if model_output == 1:
        st.header("SPAM!")
    else:
        st.header("NOT SPAM!")