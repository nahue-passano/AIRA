import pyroomacoustics as pra
from pyroomacoustics.directivities import DirectionVector, DirectivityPattern
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List
from copy import deepcopy

from aira.utils import convert_polar_to_cartesian, convert_ambisonics_a_to_b

CartesianCoordinates = namedtuple(
    "CartesianCoordinates", "x y z"
)  # not used, array preferred
AFORMAT_CAPSULES = (
    "front_left_up",
    "front_right_down",
    "back_right_up",
    "back_left_down",
)


@dataclass
class Directivity:
    azimuth: float
    altitude: float

    def to_pyroomacoustics(
        self,
    ) -> pra.directivities.Directivity:
        return pra.CardioidFamily(
            orientation=DirectionVector(
                azimuth=self.azimuth, colatitude=90 - self.altitude
            ),
            pattern_enum=pra.directivities.DirectivityPattern.CARDIOID,
        )


@dataclass
class Microphone:
    location: np.ndarray
    directivity: Directivity

    def add_to_pyroomacoustics_room(self, room: pra.Room) -> pra.Room:
        return room.add_microphone(
            loc=self.location, directivity=self.directivity.to_pyroomacoustics()
        )


@dataclass
class AmbisonicsAFormatMicrophone:
    """Ambisonics A-Format microphone."""

    location_meters: np.ndarray
    radius_cm: float
    radius_meters: float = field(init=False)
    front_left_up: Microphone = field(init=False)
    front_right_down: Microphone = field(init=False)
    back_right_up: Microphone = field(init=False)
    back_left_down: Microphone = field(init=False)

    @classmethod
    def get_aformat_capsule_directivities(
        cls,
    ) -> Dict[str, dict]:
        altitude_up = 35.26
        altitude_down = -altitude_up

        azimuth_right = 45
        azimuth_left = -azimuth_right

        return {
            "front_left_up": {"azimuth": azimuth_left, "altitude": altitude_up},
            "front_right_down": {"azimuth": azimuth_right, "altitude": altitude_down},
            "back_right_up": {"azimuth": azimuth_right + 90, "altitude": altitude_up},
            "back_left_down": {"azimuth": azimuth_left - 90, "altitude": altitude_down},
        }

    @classmethod
    def get_aformat_capsule_translations(cls, radius) -> Dict[str, np.ndarray]:
        capsule_angles = cls.get_aformat_capsule_directivities()
        return {
            capsule: np.array(
                [
                    *convert_polar_to_cartesian(
                        radius,
                        np.deg2rad(capsule_angles[capsule]["azimuth"]),
                        np.deg2rad(capsule_angles[capsule]["altitude"]),
                    )
                ]
            )
            for capsule in AFORMAT_CAPSULES
        }

    @classmethod
    def translate_aformat_capsules(cls, location: np.ndarray, radius: float):
        capsule_placements = cls.get_aformat_capsule_translations(radius)
        capsule_placements["front_left_up"] += location
        capsule_placements["front_right_down"] += location
        capsule_placements["back_right_up"] += location
        capsule_placements["back_left_down"] += location
        return capsule_placements

    def __post_init__(self):
        self.radius_meters = self.radius_cm / 100
        capsule_locations = self.translate_aformat_capsules(
            self.location_meters, self.radius_meters
        )
        directivities = self.get_aformat_capsule_directivities()
        self.front_left_up = Microphone(
            capsule_locations["front_left_up"],
            Directivity(**directivities["front_left_up"]),
        )
        self.front_right_down = Microphone(
            capsule_locations["front_right_down"],
            Directivity(**directivities["front_right_down"]),
        )
        self.back_right_up = Microphone(
            capsule_locations["back_right_up"],
            Directivity(**directivities["back_right_up"]),
        )
        self.back_left_down = Microphone(
            capsule_locations["back_left_down"],
            Directivity(**directivities["back_left_down"]),
        )

    def add_to_pyroomacoustics_room(self, room: pra.Room) -> pra.Room:
        room = self.front_left_up.add_to_pyroomacoustics_room(room)
        room = self.front_right_down.add_to_pyroomacoustics_room(room)
        room = self.back_left_down.add_to_pyroomacoustics_room(room)
        return self.back_right_up.add_to_pyroomacoustics_room(room)


desired_rt60 = 1.5  # seconds
room_dimensions = [13, 11.5, 6]  # X (front-back), Y (left-right), Z (up-down) [meters]

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(desired_rt60, room_dimensions)

# Create the room
room = pra.ShoeBox(
    room_dimensions, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
)
ambi_mic = AmbisonicsAFormatMicrophone(
    location_meters=[room_dimensions[0] - 5.9, 5.75, 2.0], radius_cm=2
)
print(ambi_mic)
room = ambi_mic.add_to_pyroomacoustics_room(room)
print(f"Microphones:\n", room.mic_array)

source_s1_location = [room_dimensions[0] - 2.3, 5.75, 3.2]
source_directivity = pra.CardioidFamily(
    orientation=DirectionVector(azimuth=-180, colatitude=0),
    pattern_enum=pra.directivities.DirectivityPattern.CARDIOID,
)
room.add_source(source_s1_location, directivity=source_directivity)

room.compute_rir()

rir_bformat = convert_ambisonics_a_to_b(
    room.rir[0][0], room.rir[1][0], room.rir[2][0], room.rir[3][0]
)

fig, axs = plt.subplots(4, 1, figsize=(18, 10))
rir_db = 20 * np.log10(rir_bformat / rir_bformat.max(axis=1))
axs[0].plot(rir_db[0, :])
axs[1].plot(rir_db[1, :])
axs[2].plot(rir_db[2, :])
axs[3].plot(rir_db[3, :])
fig.show()
