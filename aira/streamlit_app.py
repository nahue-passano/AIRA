import os
import tempfile
import streamlit as st

from aira.core import AmbisonicsImpulseResponseAnalyzer
from aira.engine.input import InputMode


def save_temp_file(file):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file.getvalue())
    return temp_file_path


def run_streamlit_app():
    st.set_page_config(
        page_title="AIRA", page_icon="docs/images/aira-icon.png", layout="wide"
    )

    # Logo
    logo_path = "docs/images/aira-banner.png"
    personal_logo_path = "docs/images/nahue-passano.png"
    st.columns(11)[10].image(personal_logo_path, use_column_width=True)

    # Audio loading and settings

    settings, audio_files = st.columns(2)

    with settings:
        st.image(logo_path, use_column_width=True)
        st.subheader("⚙️ Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            integration_time = st.selectbox("Integration time [ms]", [1, 5, 10])
        with col2:
            analysis_length = st.text_input("Analysis length [ms]", value="500")
        with col3:
            intensity_threshold = st.text_input("Intensity threshold [dB]", value=-60)

    with audio_files:
        st.subheader("🔉 LSS room responses in A-Format")
        up_files, down_files = st.columns(2)

        with up_files:
            audio_file_flu = st.file_uploader("Front-Left-Up", type=["mp3", "wav"])
            audio_file_bru = st.file_uploader("Back-Right-Up", type=["mp3", "wav"])

        with down_files:
            audio_file_frd = st.file_uploader("Front-Right-Down", type=["mp3", "wav"])
            audio_file_bld = st.file_uploader("Back-Left-Down", type=["mp3", "wav"])

        audio_file_inverse_filter = st.file_uploader(
            "Inverse filter", type=["mp3", "wav"]
        )

    # "Analyze" button
    if st.button("Analyze", use_container_width=True):
        data = {
            "front_left_up": save_temp_file(audio_file_flu),
            "front_right_down": save_temp_file(audio_file_frd),
            "back_right_up": save_temp_file(audio_file_bru),
            "back_left_down": save_temp_file(audio_file_bld),
            "inverse_filter": save_temp_file(audio_file_inverse_filter),
            "input_mode": InputMode.LSS,
            "channels_per_file": 1,
            "frequency_correction": True,
        }
        analyzer = AmbisonicsImpulseResponseAnalyzer()
        fig = analyzer.analyze(
            input_dict=data,
            integration_time=float(integration_time) / 1000,
            intensity_threshold=float(intensity_threshold),
            analysis_length=float(analysis_length) / 1000,
            show=False,
        )
        fig.update_layout(
            height=1080,
            paper_bgcolor="rgb(14,17,23)",
            plot_bgcolor="rgb(14,17,23)",
        )
        st.plotly_chart(fig, use_container_width=True, height=1080)


if __name__ == "__main__":
    run_streamlit_app()
