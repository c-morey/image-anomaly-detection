import streamlit as st
from PIL import Image
from cumulative_images import cumulative_images
from find_defect import find_defect

@st.cache(allow_output_mutation=True)
def createExtractor():
     return cumulative_images(), find_defect()

def process(uploaded_files):
    i=1
    for uploaded_file in uploaded_files:
        my_expander = st.expander(label=uploaded_file.name)
        my_expander.write('Hello there!')
        clicked = my_expander.button('Click me!', key=i)
        i+=1
    
# workExpExtractor, sectionExtractor = createExtractor()

#st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Image Anomaly Detection App")
st.markdown( "This app aims to detect **anomaly dice images.**")
st.markdown("For more info: [Check GitHub](https://github.com/c-morey/image-anomaly-detection)")

# Main page image
# img = Image.open("./streamlit_images/st_dice_image.png")
# st.image(img)

# Creating a side bar for users to explore
st.sidebar.markdown("## Side Bar")
st.sidebar.markdown("Use this panel to explore the details of images, and find the defected ones.")

#st.title('Create Cumulative Images')
# At this step show the contour number and images of how they look like
st.sidebar.subheader('Create Cumulative Images')
st.sidebar.markdown("Choose the images you want to create a cumulative image.")
# uploaded_file = st.file_uploader("Please upload your images here",type=['png','jpeg', 'jpg'])
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

if st.button('test'):
    process(uploaded_files)

if st.button('train'):
    cumulative_images()

# Show the matplotlib viz of avg and std - its before setting the final margin
st.sidebar.subheader('Train the Images')
st.sidebar.markdown("This is an optional step. You can train your dataset before checking for anomalies.")
# TODO: add sort of confirmation button

# Final step -- show the image with final margin matplotlib and the name of the folders found as anomaly
st.sidebar.subheader('Detect Anomalies')
st.sidebar.markdown("This is the final step to find the anomaly images.")
# TODO: show which images are anomaly
