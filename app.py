import streamlit as st
from agent.global_agent import *
from agent.mcp_client import *
from agent.helper_functions import *
import os

class StreamlitUIUpdater:
    def __init__(self):
        st.write("# Welcome!")
        st.write(f"Song that will be playing:{path}")

    def mainloop(self, model = model):
        # initialize session variable `messages` for displaying messages
        if "messages" not in st.session_state:
            st.session_state.messages = []


        # initialize session variable `show_audio_player` for displaying audio player
        if "show_audio_player" not in st.session_state:
            st.session_state.show_audio_player = False

        # checks the state of the audio player and sets it
        if get_state() == True:
            print("Audio player is ON")
            st.session_state.show_audio_player = True
            set_state(True)
        else:
            print("Audio player is OFF")
            st.session_state.show_audio_player = False
            set_state(False)

        # if the audio player has to be visible, make it visible:
        try:
            if st.session_state.show_audio_player:
                st.audio('./' + path, format='audio/wav', autoplay=True, loop=True)
        except Exception as e:
            st.write("Error displaying audio player:", e)
            st.session_state.show_audio_player = False
            set_state(False)

        # print messages from chat history (i.e., the `messages` variable)
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.write(message['content'])

        # activates the query-response system upon user prompt
        if query := st.chat_input("What's on your mind? Drop it off in here :)", key = "Input"):

            # step 1: make markdown version of user query and add it to chat history
            st.chat_message("user").markdown(query)
            st.session_state.messages.append({"role": "user", "content":query})
            

            # step 2: generate response
            response = model.run_once(query)['assistant']
            # step 3: add response to chat history
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role":"assistant","content":response})

if __name__ == "__main__":
    app = StreamlitUIUpdater()
    try:
        app.mainloop(model=model)
    except Exception as e:
        st.write(e.args)
    set_state(False)