from app_def import *
import streamlit as st

if __name__ == "__main__":
    try:
        app.mainloop(model=model)
    except Exception as e:
        st.write(e.args)