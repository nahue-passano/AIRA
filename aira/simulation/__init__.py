from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import pyroomacoustics as pra
from pyroomacoustics.directivities import DirectionVector
import matplotlib.pyplot as plt
import numpy as np

from aira.utils import convert_polar_to_cartesian

CartesianCoordinates = namedtuple(
    "CartesianCoordinates", "x y z"
)  # not used, array preferred

# TODO: replace every use of the tuple with `AFormatCapsules`. The latter needs to have the sorting implemented.
AFORMAT_CAPSULES = (
    "front_left_up",
    "front_right_down",
    "back_right_up",
    "back_left_down",
)


class AFormatCapsules(Enum):
    """The four A-Format Ambisonics capsules."""

    FRONT_LEFT_UP = "front_left_up"
    FRONT_RIGHT_DOWN = "front_right_down"
    BACK_RIGHT_UP = "back_right_up"
    BACK_LEFT_DOWN = "back_left_down"


@dataclass
class Directivity:
    """Class for simplifying the interface of a directivity pattern."""

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
    """Class for storing details of a microphone: location and directivity."""

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
    sampling_rate_pyroomacoustics: int = field(default=44100)
    radius_meters: float = field(init=False)
    front_left_up: Microphone = field(init=False)
    front_right_down: Microphone = field(init=False)
    back_right_up: Microphone = field(init=False)
    back_left_down: Microphone = field(init=False)

    @classmethod
    def get_aformat_capsule_directivities(
        cls,
    ) -> Dict[str, dict]:
        """Get the directivity details of each of the A-Format cardioid capsules.

        Returns:
            Dict[str, dict]: a dictionary with each of the A-Format capsule names
                as keys, and the values are dictionaries with keys `azimuth` and
                `altitude`.
        """
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
    def get_aformat_capsule_translations(cls, radius: float) -> Dict[str, np.ndarray]:
        """Get the required translation vector for each capsule of an A-Format
        Ambisonics microphone with the given radius. The radius is the distance
        of each capsule to the center of the array.

        Args:
            radius (float): radius in meters.

        Returns:
            Dict[str, np.ndarray]: a dictionary with A-Format capsule names as keys and
                NumPy arrays as values, with the translation vector coordinates (x,y,z)
                (i.e., arrays of shape (3,)).
        """
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
    def translate_aformat_capsules(
        cls, location: np.ndarray, radius: float
    ) -> Dict[str, np.ndarray]:
        """Move each capsule in the microphone based on the reported radius.

        Args:
            location (np.ndarray): location of the center of the microphone array.
            radius (float): distance from each capsule to the center of the array, in meters.

        Returns:
            Dict[str, np.ndarray]: a dictionary with each of the A-Format capsule names as keys
                and a NumPy array with the capsule coordinates (x, y, z) as values, with shape (3,).
        """
        capsule_placements = cls.get_aformat_capsule_translations(radius)
        capsule_placements["front_left_up"] += location
        capsule_placements["front_right_down"] += location
        capsule_placements["back_right_up"] += location
        capsule_placements["back_left_down"] += location
        return capsule_placements

    def __post_init__(self):
        """Compute each of the four microphones' characteristics."""
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
        """Add each of the microphones to the PyRoomAcoustics Room. It doesn't work because
        PyRoomAcoustics forget to load the directivities into the equivalent MicrophoneArray.
        Use the method `self.to_pyroomacoustics_array()` instead, and add the MicrophoneArray
        to the Room manually.

        Args:
            room (pra.Room): PyRoomAcoustics Room.

        Returns:
            pra.Room: modified Room with each of the microphones loaded.
        """
        room = self.front_left_up.add_to_pyroomacoustics_room(room)
        room = self.front_right_down.add_to_pyroomacoustics_room(room)
        room = self.back_left_down.add_to_pyroomacoustics_room(room)
        return self.back_right_up.add_to_pyroomacoustics_room(room)

    def get_capsule(self, capsule: AFormatCapsules) -> Microphone:
        """Get the microphone corresponding to the given A-Format capsule.

        Arguments
            capsule (AFormatCapsule): the Enum for the A-Format capsule.
        Returns
            (Microphone): an object with information about its location and directivity.
        """
        try:
            return self.__getattribute__(capsule.value)
        except ValueError as exception:
            raise ValueError("%s is not a valid A-Format capsule name" % capsule.value)

    def to_pyroomacoustics_array(
        self,
    ) -> pra.MicrophoneArray:
        """Cast the object into a PyRoomAcoustics MicrophoneArray, including cardioid directivities.

        Returns
            pyroomacoustics.MicrophoneArray
        """
        locations = np.zeros((3, 4))  # 3 coordinates, 4 microphones
        directivities = []
        for i, capsule in enumerate(AFormatCapsules):
            microphone = self.get_capsule(capsule)
            locations[:, i] = microphone.location
            directivities.append(microphone.directivity.to_pyroomacoustics())

        return pra.MicrophoneArray(
            locations, fs=self.sampling_rate_pyroomacoustics, directivity=directivities
        )


def plot_rir(rir_array: np.ndarray):
    """Plot a RIR array.

    Arguments
        rir_array (np.ndarray): a (4, N)-shaped RIR array.
    """
    fig, axs = plt.subplots(4, 1, figsize=(18, 10))
    axs[0].plot(rir_array[0, :])
    axs[1].plot(rir_array[1, :])
    axs[2].plot(rir_array[2, :])
    axs[3].plot(rir_array[3, :])
    plt.show()
