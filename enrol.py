import streamlit as st
from PIL import Image
import os
import utils 
import json

def remove_data(candidate_id):

    data_dir = './database'
    with open(data_dir+"/data.json", "r") as f:
        data = json.load(f)

    try:
        idx = data['ids'].index(candidate_id)
        data['encodings'].pop(idx)
        data['ids'].pop(idx)
        data['names'].pop(candidate_id)

        with open(data_dir+"/data.json", "w") as f:
            json.dump(data, f)
    except:
        return False

    return True


def save_image(image, file_name):

    save_dir = './database/images'
    img = Image.open(image)
    save_path = os.path.join(save_dir, file_name+'.jpg')

    img.save(save_path)

    return save_path, img


def enrol_candidate():
    st.title('Enroll/Remove A Candidate')

    tab1, tab2 = st.tabs(['Enrol', 'Remove'])

    with tab1:
        name = st.text_input('Enter Name')
        id = st.text_input('Enter Unique Id/Roll No')

        input_preference = st.selectbox('Choose Input Type', options=['Files', 'Camera'])
        usable_img = None

        if input_preference=='Camera':
            cam_feed = st.camera_input('Open Camera')
            if cam_feed is not None:
                usable_img = cam_feed
        
        else:
            uploaded_img = st.file_uploader('Upload a recent image')
            if uploaded_img is not None:
                usable_img = uploaded_img

        if usable_img:
            save_path, img = save_image(usable_img, id)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            if st.button('Add Data'):
                ret = utils.encode_image(save_path, id, name)
                if ret:
                    st.success('Data Added Successfully!')

    with tab2:
        candidate_id = st.text_input('Enter Candidate Id')
        choice = st.checkbox('Remove all encoding details of this ID')
        if choice and st.button('Remove Data'):
            ret = remove_data(candidate_id)
            if ret:
                st.success(f'Details of {candidate_id} are removed successfully')
            else:
                st.error(f'No Data found for {candidate_id}')
