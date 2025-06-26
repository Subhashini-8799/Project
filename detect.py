import streamlit as st
import utils
import pandas as pd


def display_csv(csv_file='./database/record.csv'):
    try:
        data = pd.read_csv(csv_file)
        st.dataframe(data, use_container_width=True)
    
    except pd.errors.EmptyDataError:

        st.write("The CSV file is empty or does not exist yet.")


def video_processing():
    # st.title('Logging In And Out Surveillance')
    choice = st.selectbox('Choose Input Stream', ['Upload a Video', 'Live Video Feed'])
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "asf", "m4v"])
    ret = False

    id_data = []

    if st.button('Stream'):
        if choice == 'Live Video Feed':
            ret = utils.detect_faces(0, id_data)
        elif choice == 'Upload a Video':
            if uploaded_file is not None:
                # Read video file
                with open('input.mp4', 'wb') as uf:
                    uf.write(uploaded_file.getbuffer())

                ret = utils.detect_faces('./input.mp4', id_data)
        
        if ret:
            display_csv()


# if __name__=='__main__':
#     video_processing()