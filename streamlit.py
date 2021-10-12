import streamlit as st
from PIL import Image
from cumulative_images import generate_all_cumulative_images
from find_defect import is_defect


def process(uploaded_files):
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        expander_title = uploaded_file.name
        is_defective, face_number = is_defect(bytes_data)
        if is_defective:
            expander_title += " - Anomalous die"
        else:
            expander_title += " - Correct die"

        my_expander = st.expander(label=expander_title)
        image = Image.open(uploaded_file)
        binary_dice = Image.open(
            f"data/preprocessed_dice_images/binary_dice_{face_number}.png")
        contour_dice = Image.open(
            f"data/preprocessed_dice_images/contour_dice_{face_number}.png")
        cumulative_dice = Image.open(
            f"data/preprocessed_dice_images/cumulative_dice_{face_number}.png")
        my_expander.image(image, width=256)
        my_expander.image([cumulative_dice, binary_dice, contour_dice], width=220)


st.title("Image Anomaly Detection App")
st.markdown("This app aims to detect **anomaly dice images.**")
st.markdown("For more info: [Check GitHub](https://github.com/c-morey/image-anomaly-detection)")

# Creating a side bar for users to explore
st.sidebar.markdown("## Side Bar")
st.sidebar.markdown("Use this panel to explore the details of images, and find the defected ones.")

# At this step show the contour number and images of how they look like
st.sidebar.subheader('Create Cumulative Images')
st.sidebar.markdown("Choose the images you want to create a cumulative image.")
uploaded_files = st.file_uploader("Please upload your images here", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)
if st.sidebar.button('Generate Cumulative Images'):
    generate_all_cumulative_images()

# Final step -- show the image with final margin matplotlib and the name of the folders found as anomaly
st.sidebar.subheader('Detect Anomalies')
st.sidebar.markdown("This is the final step to find the anomaly images.")
if st.sidebar.button('Check anomaly'):
    process(uploaded_files)
