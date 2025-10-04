import streamlit as st
import requests

API_URL = "http://localhost:8000"  


def detect_image_ui():
    st.header("📸 Detect on Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Image"):
            files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
            response = requests.post(f"{API_URL}/detect/image_bbox/", files=files)
            if response.status_code == 200:
                st.image(response.content, caption="Detected Image", use_column_width=True)
            else:
                st.error("❌ Error detecting image")


def detect_video_ui():
    st.header("🎥 Detect on Video")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video")

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Detect Video"):
            with st.spinner("⏳ Đang xử lý video..."):
                uploaded_video.seek(0)
                files = {"file": (uploaded_video.name, uploaded_video, "video/mp4")}
                response = requests.post(f"{API_URL}/detect-video", files=files)

                if response.status_code == 200:
                    video_bytes = response.content
                    st.success("✅ Video processed successfully!")
                    st.video(video_bytes) 
                else:
                    st.error("❌ Error detecting video")


def main():
    st.set_page_config(page_title="Helmet Detection", layout="centered")
    st.title("🪖 Helmet Detection Demo")
    st.write("Upload image or video to detect helmets using YOLOv8")

    detect_image_ui()
    detect_video_ui()


if __name__ == "__main__":
    main()
