"""Unit tests for the audio utility functions module."""

from mock_data.recordings import (  # pylint: disable=unused-import
    aformat_signal_and_samplerate,
)

from aira.utils import read_aformat


def test_read_aformat_from_list(
    aformat_signal_and_samplerate,
):  # pylint: disable=redefined-outer-name
    """WHEN trying to read an A-format Ambisonics recording, GIVEN the channels
    are passed as a list to the reading function, THEN call the corresponding
    overloaded function.

    Args:
        aformat_signal_and_samplerate (tuple): a PyTest fixture which returns
        a tuple with an array of A-format Ambisonics signals (1 row per channel)
        and their sample rate in the second element of the tuple.
    """
    (
        expected_signal,
        expected_sample_rate,
    ) = aformat_signal_and_samplerate  # pylint: disable=unpacking-non-sequence
    expected_shape = expected_signal.shape

    audio_path_list = [
        "./test/mock_data/regio_theater/soundfield_flu.wav",
        "./test/mock_data/regio_theater/soundfield_frd.wav",
        "./test/mock_data/regio_theater/soundfield_bru.wav",
        "./test/mock_data/regio_theater/soundfield_bld.wav",
    ]
    audio_array, sample_rate = read_aformat(  # pylint: disable=unpacking-non-sequence
        audio_path_list
    )

    assert (
        expected_shape == audio_array.shape
    ), f"Output shape: {audio_array.shape} != Expected shape: {expected_shape}"
    assert (
        expected_sample_rate == sample_rate
    ), f"Output sample rate: {sample_rate} != Expected sample rate: {expected_sample_rate}"


def test_read_aformat_from_dict(
    aformat_signal_and_samplerate,
):  # pylint: disable=redefined-outer-name
    """WHEN trying to read an A-format Ambisonics recording, GIVEN the channels
    are passed as a dictionary to the reading function, THEN call the
    corresponding overloaded function.

    Args:
        aformat_signal_and_samplerate (tuple): a PyTest fixture which returns
        a tuple with an array of A-format Ambisonics signals (1 row per channel)
        and their sample rate in the second element of the tuple.
    """
    (
        expected_signal,
        expected_sample_rate,
    ) = aformat_signal_and_samplerate  # pylint: disable=unpacking-non-sequence
    expected_shape = expected_signal.shape

    audio_paths = dict(  # pylint: disable=use-dict-literal
        front_left_up="./test/mock_data/regio_theater/soundfield_flu.wav",
        front_right_down="./test/mock_data/regio_theater/soundfield_frd.wav",
        back_right_up="./test/mock_data/regio_theater/soundfield_bru.wav",
        back_left_down="./test/mock_data/regio_theater/soundfield_bld.wav",
    )
    audio_array, sample_rate = read_aformat(  # pylint: disable=unpacking-non-sequence
        audio_paths
    )

    assert (
        expected_shape == audio_array.shape
    ), f"Output shape: {audio_array.shape} != Expected shape: {expected_shape}"
    assert (
        expected_sample_rate == sample_rate
    ), f"Output sample rate: {sample_rate} != Expected sample rate: {expected_sample_rate}"
