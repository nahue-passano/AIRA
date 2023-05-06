from aira.utils import read_aformat
import soundfile as sf


def test_read_audio():
    audio_path_1 = "test/mock_data/soundfield_measurements/soundfield_bld.wav"
    audio_path_2 = "test/mock_data/soundfield_measurements/soundfield_bru.wav"
    audio_path_list = [audio_path_1, audio_path_2, "", ""]

    # Expected values
    expected_audio_1, _ = sf.read(audio_path_1)
    expected_audio_2, expected_sample_rate = sf.read(audio_path_2)
    expected_audio_array_shape = (len(audio_path_list), len(expected_audio_2))

    audio_array, sample_rate = read_aformat(audio_path_list)

    assert (
        expected_audio_array_shape == audio_array.shape
    ), f"Output shape: {audio_array.shape} != Expected shape: {expected_audio_array_shape}"
    assert (
        expected_sample_rate == sample_rate
    ), f"Output sample rate: {sample_rate} != Expected sample rate: {expected_sample_rate}"
