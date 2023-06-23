"""Example usage of PyRoomAcoustics with AIRA A-Format microphone array.
The image source method (ISM) simulation will run after plotting the room.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

from aira.utils import convert_ambisonics_a_to_b
from aira.simulation import AmbisonicsAFormatMicrophone, plot_rir

# Room design
SAMPLING_RATE = 16000
desired_rt60 = 1.5  # seconds
room_dimensions = [13, 11.5, 6]  # X (front-back), Y (left-right), Z (up-down) [meters]

# Invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(desired_rt60, room_dimensions)

# Create the room
room = pra.ShoeBox(
    room_dimensions,
    fs=SAMPLING_RATE,
    materials=pra.Material(e_absorption),
    max_order=max_order,
)

# Design the A-Format microphone
ambi_mic = AmbisonicsAFormatMicrophone(
    location_meters=[room_dimensions[0] - 5.9, 5.75, 2.0],
    radius_cm=2,
    sampling_rate_pyroomacoustics=SAMPLING_RATE,
)
print(ambi_mic)

room = room.add_microphone_array(
    ambi_mic.to_pyroomacoustics_array()
)  # this way i force the directivity

source_s1_location = [room_dimensions[0] - 2.3, 5.75, 3.2]
source_directivity = pra.CardioidFamily(
    orientation=pra.DirectionVector(azimuth=-180, colatitude=0),
    pattern_enum=pra.directivities.DirectivityPattern.CARDIOID,
)
room.add_source(source_s1_location, directivity=source_directivity)
room.plot()
plt.show()

room.compute_rir()

max_rir_length: int = 0
for rir in room.rir:
    if len(rir[0]) > max_rir_length:
        max_rir_length = len(rir[0])
for i, rir in enumerate(room.rir):
    padded_array = np.zeros((max_rir_length,))
    if len(rir[0]) < max_rir_length:
        print(f"RIR number {i} is shorter than {max_rir_length = }")
        padded_array[: len(rir[0])] = rir[0]
        rir[0] = padded_array

rir_bformat = convert_ambisonics_a_to_b(
    room.rir[0][0], room.rir[1][0], room.rir[2][0], room.rir[3][0]
)

# Awful moving average filtering for smoothing the RIR.
rir_smoothed = np.zeros_like(rir_bformat)
for i in range(max_rir_length):
    rir_smoothed[:, i] = rir_bformat[:, i : i + 5].mean(axis=1)
rir_smoothed

rir_db = 20 * np.log10(np.abs(rir_smoothed) / np.abs(rir_bformat).max() + 0.001)
plot_rir(rir_db)
