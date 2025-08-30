import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

# import your existing functions from webmorph.py
from newway import videopart1, videosubtraction, videopart2, morphing, concat_videos

st.title("Body Morphing Web App")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
gap = st.number_input("Enter gap interval time (seconds)", min_value=0, value=1)

if uploaded_file is not None:
    # Save uploaded video temporarily
    tfile = NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("Run Pipeline"):
        trimmed = videopart1(video_path)
        remaining = videosubtraction(gap, video_path, trimmed, "final_recording_matched.mp4")
        recorded = videopart2(remaining)
        morph = morphing("gesture_1.jpg", "gesture_2.jpg")
        output = concat_videos([trimmed, morph, recorded], "merged_result.mp4")

        st.success(" Pipeline complete!")
        st.video("merged_result.mp4")
