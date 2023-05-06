import soundfile as sf

from aira.utils import read_aformat
from mock_data.recordings import load_mocked_aformat


def test_read_aformat_from_list():
    expected_signal, expected_sample_rate = load_mocked_aformat()
    expected_shape = expected_signal.shape

    audio_path_list = [
        "./test/mock_data/soundfield_flu.wav",
        "./test/mock_data/soundfield_frd.wav",
        "./test/mock_data/soundfield_bru.wav",
        "./test/mock_data/soundfield_bld.wav",
    ]
    audio_array, sample_rate = read_aformat(audio_path_list)

    assert (
        expected_shape == audio_array.shape
    ), f"Output shape: {audio_array.shape} != Expected shape: {expected_shape}"
    assert (
        expected_sample_rate == sample_rate
    ), f"Output sample rate: {sample_rate} != Expected sample rate: {expected_sample_rate}"


def test_read_aformat_from_dict():
    expected_signal, expected_sample_rate = load_mocked_aformat()
    expected_shape = expected_signal.shape

    audio_paths = dict(
        front_left_up="./test/mock_data/soundfield_flu.wav",
        front_right_down="./test/mock_data/soundfield_frd.wav",
        back_right_up="./test/mock_data/soundfield_bru.wav",
        back_left_down="./test/mock_data/soundfield_bld.wav",
    )
    audio_array, sample_rate = read_aformat(audio_paths)

    assert (
        expected_shape == audio_array.shape
    ), f"Output shape: {audio_array.shape} != Expected shape: {expected_shape}"
    assert (
        expected_sample_rate == sample_rate
    ), f"Output sample rate: {sample_rate} != Expected sample rate: {expected_sample_rate}"

