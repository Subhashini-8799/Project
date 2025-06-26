import streamlit as st
import json
import os
import base64
import enrol 
import detect

def home():

    col1, col2 = st.columns(2)

    with col1:
        st.success('Enrol a New Candidate:')
        st.write('To add a new candidate to the existing data or Remove a data from the database')
        st.success('Stream Video Feed:')
        st.write('To perform real-time detection on live or uploaded video feed and visualize the recorded metrics')
        st.write('From the selectbox choose live/Upload video for streaming')
        st.write('Display data to visualize the recorded dataframe')
        st.warning('Note: Enrol atleast one candidate before choosing Stream Video Feed')
       

    with col2:
        file_ = open('./assets/face-2.gif', "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )


def main():

    st.set_page_config(layout='wide')
    st.title('Logging In And Out Surveillance')

    pages = {
        "Home": "Home",
        "Enrol New Candidate": "Enrol",
        "Stream Video Feed": "Stream"
    }

    selected_page = st.sidebar.selectbox("Navigation", list(pages.keys()))

    if pages[selected_page]=='Home':
        home()
        
    elif pages[selected_page]=='Enrol':
        enrol.enrol_candidate()

    elif pages[selected_page]=='Stream':
        detect.video_processing()

    


if __name__=='__main__':
    main()