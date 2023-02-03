import streamlit as st
import requests


def main():

    new_title = '<p style="font-size: 42px;">' \
                'Object Detection with Voice Feedback App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    This project was built using Streamlit and fastAPI to demonstrate 
    YOLOV7 Object detection videos - pre-recorded and streaming 
    (to be built), then translate the predicted objects into voice 
    using google Text-to-Speech API. This project is built in the 
    aim of helping visually impaired person to 'see' better the 
    obstacles when they are moving around. 

    The YOLO object detector was prepaired base on the offical 
    yolov7 tiny model with pretrained weights, then retrained 
    with custom dataset. 
    Below are some useful links: 

    [yolov7 official github](https://github.com/WongKinYiu/yolov7)  
    
    [dataset used for re-training](https://www.nuscenes.org/download)
    """)
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox("MODE", ("About", "Object Detection(Video)"))

    if choice == "Object Detection(Video)":
        read_me_0.empty()
        read_me.empty()

        uploaded_video = st.file_uploader("Upload Video",
                                          type=['mp4', 'mpeg', 'mov', 'txt'])
        st.video(uploaded_video)
        if uploaded_video != None:

            # Submit Button
            submit_file = st.button("Submit")
            file = {'file':(uploaded_video.name, uploaded_video)}

            # On submit events
            if submit_file:
                response = requests.post("http://127.0.0.1:8000/upload",
                                         files=file)
                st.write(response.status_code)

                # display the predicted video
                st.write('The output video with objects detected')
                predicted_video = open(response.text, 'rb')
                st.video(predicted_video)

    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
