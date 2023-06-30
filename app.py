import streamlit as st
import plotly.graph_objects as go
from aira.core import AmbisonicsImpulseResponseAnalyzer
from aira.engine.input import InputMode
import os
import tempfile


def save_temp_file(file):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file.getvalue())
    return temp_file_path


def main():
    st.set_page_config(
        page_title="AIRA", page_icon="docs/images/aira-icon.png", layout="wide"
    )

    # Logo
    logo_path = "docs/images/aira-banner.png"
    personal_logo_path = "docs/images/nahue-passano.png"
    st.columns(11)[10].image(personal_logo_path, use_column_width=True)
    st.columns(3)[1].image(logo_path, use_column_width=True)

    # Cuadros de carga de archivos de audio
    st.header("🔉 LSS room responses in A-Format")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        audio_file_flu = st.file_uploader("Front-Left-Up", type=["mp3", "wav"])
    with col2:
        audio_file_frd = st.file_uploader("Front-Right-Down", type=["mp3", "wav"])
    with col3:
        audio_file_bru = st.file_uploader("Back-Right-Up", type=["mp3", "wav"])
    with col4:
        audio_file_bld = st.file_uploader("Back-Left-Down", type=["mp3", "wav"])
    with col5:
        audio_file_inverse_filter = st.file_uploader(
            "Inverse filter", type=["mp3", "wav"]
        )

    # Configuraciones
    st.header("⚙️ Settings")
    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        integration_time = st.selectbox("Integration time [ms]", [1, 5, 10])
    with config_col2:
        analysis_length = st.text_input("Analysis length [ms]", value="500")
    with config_col3:
        intensity_threshold = st.text_input("Intensity threshold [dB]", value=-60)

    # Botón "Analyze"
    if st.columns(3)[1].button("Analyze", use_container_width=True):
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

        analyzer = AmbisonicsImpulseResponseAnalyzer(
            int(integration_time), float(intensity_threshold), float(analysis_length)
        )
        fig = analyzer.analyze(data, show=False)
        fig.update_layout(height=1080)
        st.plotly_chart(fig, use_container_width=True, height=1080)

        # Generar un gráfico genérico con Plotly
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 5, 7, 2]))
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
