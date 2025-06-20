
# Mee Streamlit code ikkada paste cheyyandi

import streamlit as st

st.title("My First Web App")

number = st.number_input("Enter a number:", value=0)

st.write(f"The square of {number} is {number ** 2}")
