import streamlit as st
import time
from utils import chat
from ui import hide_streamlit_style, header

st.set_page_config(page_title="Med Assist Chat", page_icon="🩺", layout="wide")
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
header()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessionid" not in st.session_state:
    st.session_state.sessionid = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("You can ask any question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if st.session_state.sessionid is None:
            with st.spinner():
                data = chat(prompt, session_id=None)
                assistant_response = data["output"]
            st.session_state.sessionid = data["metadata"]["run_id"]
        else:
            with st.spinner():
                data = chat(prompt, session_id=st.session_state.sessionid)
                assistant_response = data["output"]
            print(assistant_response)

        message_placeholder = st.empty()
        full_response = ""
        # # Simulate stream of response with milliseconds delay
        # if "So you are facing" not in assistant_response:
        #     for chunk in assistant_response.split():
        #         full_response += chunk + "\n"
        #         time.sleep(0.05)
        #         # Add a blinking cursor to simulate typing
        #         message_placeholder.markdown(full_response + "▌")
        #     message_placeholder.markdown(full_response)
        # elif "Possible Legal Strategies" in assistant_response:
        #     full_response = assistant_response
        #     message_placeholder.markdown(full_response + "▌")

        full_response = assistant_response
        message_placeholder.write(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
