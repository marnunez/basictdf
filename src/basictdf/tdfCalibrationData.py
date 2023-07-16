from enum import IntEnum
from typing import List, Union

import numpy as np

from basictdf.tdfBlock import Block, BlockType, BuildWriteable, Sized
from basictdf.tdfTypes import (
    MAT3X3D,
    MAT3X3F,
    VEC2D,
    VEC3D,
    VEC3F,
    CameraViewPort,
    TdfType,
    f64,
    i16,
    i32,
)

__doc__ = "The TDF Calibration Data block"

CameraMap = TdfType(np.dtype("(1,)<i2"))


class DistorsionModel(IntEnum):
    noDistorsion = 0
    "No distorsion"
    KaliDistorsion = 1
    "Kali distorsion (from BTS) type"
    AmassDistorsion = 2
    "Amass distorsion (from BTS) type"
    Seelab1Distorsion = 3
    "Radial distorsion up to 2nd order"


class SeelabCameraData(Sized, BuildWriteable):
    def __init__(
        self,
        rotation_matrix,
        translation_vector,
        focus,
        optical_center,
        radial_distortion,
        decentering,
        thin_prism,
        view_port: Union[CameraViewPort, np.ndarray],
    ) -> None:
        if not isinstance(rotation_matrix, np.ndarray) or rotation_matrix.shape != (
            3,
            3,
        ):
            raise TypeError("rotation_matrix must be a (3,3) shape numpy array")

        self.rotation_matrix = rotation_matrix
        "Rotation matrix of the camera"

        if not isinstance(
            translation_vector, np.ndarray
        ) or translation_vector.shape != (3,):
            raise TypeError("translation_vector must be a (3,) shape numpy array")

        self.translation_vector = translation_vector
        "Translation vector of the camera"

        if not isinstance(focus, np.ndarray) or focus.shape != (2,):
            raise TypeError("focus must be a (2,) shape numpy array")

        self.focus = focus
        "Focal length of the camera"

        if not isinstance(optical_center, np.ndarray) or optical_center.shape != (2,):
            raise TypeError("optical_center must be a (2,) shape numpy array")

        self.optical_center = optical_center
        "Optical center of the camera"

        if not isinstance(radial_distortion, np.ndarray) or radial_distortion.shape != (
            2,
        ):
            raise TypeError("radial_distortion must be a (2,) shape numpy array")

        self.radial_distortion = radial_distortion
        "Radial distortion of the camera"

        if not isinstance(decentering, np.ndarray) or decentering.shape != (2,):
            raise TypeError("decentering must be a (2,) shape numpy array")

        self.decentering = decentering
        "Decentering of the camera"

        if not isinstance(thin_prism, np.ndarray) or thin_prism.shape != (2,):
            raise TypeError("thin_prism must be a (2,) shape numpy array")

        self.thin_prism = thin_prism
        "Thin prism of the camera"

        if isinstance(view_port, CameraViewPort):
            view_port = view_port
        elif isinstance(view_port, np.ndarray) and view_port.shape == (
            2,
            2,
        ):
            view_port = CameraViewPort(view_port[0], view_port[1])
        else:
            raise TypeError(
                "view_port must be a CameraViewPort or a (2,2) shape numpy array"
            )

        self.view_port = view_port
        "Camera viewport"

    @staticmethod
    def _build(stream) -> "SeelabCameraData":
        rotation_matrix = MAT3X3D.bread(stream)
        translation_vector = VEC3D.bread(stream)
        focus = VEC2D.bread(stream)
        optical_center = VEC2D.bread(stream)
        radial_distorion = VEC2D.bread(stream)
        decentering = VEC2D.bread(stream)
        thin_prism = VEC2D.bread(stream)
        view_port = CameraViewPort.bread(stream)
        return SeelabCameraData(
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            focus=focus,
            optical_center=optical_center,
            radial_distortion=radial_distorion,
            decentering=decentering,
            thin_prism=thin_prism,
            view_port=view_port,
        )

    def _write(self, file) -> None:
        MAT3X3D.bwrite(file, self.rotation_matrix)
        VEC3D.bwrite(file, self.translation_vector)
        VEC2D.bwrite(file, self.focus)
        VEC2D.bwrite(file, self.optical_center)
        VEC2D.bwrite(file, self.radial_distortion)
        VEC2D.bwrite(file, self.decentering)
        VEC2D.bwrite(file, self.thin_prism)
        self.view_port.bwrite(file)

    @property
    def nBytes(self) -> int:
        return (
            MAT3X3D.btype.itemsize  # rotation_matrix
            + VEC3D.btype.itemsize  # translation_vector
            + VEC2D.btype.itemsize  # focus
            + VEC2D.btype.itemsize  # optical_center
            + VEC2D.btype.itemsize  # radial_distortion
            + VEC2D.btype.itemsize  # decentering
            + VEC2D.btype.itemsize  # thin_prism
            + CameraViewPort.nBytes  # view_port
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SeelabCameraData):
            return False
        return (
            np.array_equal(self.rotation_matrix, o.rotation_matrix)
            and np.array_equal(self.translation_vector, o.translation_vector)
            and np.array_equal(self.focus, o.focus)
            and np.array_equal(self.optical_center, o.optical_center)
            and np.array_equal(self.radial_distortion, o.radial_distortion)
            and np.array_equal(self.decentering, o.decentering)
            and np.array_equal(self.thin_prism, o.thin_prism)
            and self.view_port == o.view_port
        )


class BTSCameraData:
    max_distorsion_coefficients = 70

    def __init__(
        self,
        rotation_matrix,
        translation_vector,
        focus,
        optical_center,
        x_distortion_coefficients,
        y_distortion_coefficients,
        view_port: Union[CameraViewPort, np.ndarray],
    ) -> None:
        self.rotation_matrix = rotation_matrix
        "Rotation matrix of the camera"

        self.translation_vector = translation_vector
        "Translation vector of the camera"

        self.focus = focus
        "Focal length of the camera"

        self.optical_center = optical_center
        "Optical center of the camera"

        if len(x_distortion_coefficients) > self.max_distorsion_coefficients:
            raise ValueError(
                (
                    f"Can't have more than {self.max_distorsion_coefficients}"
                    " distortion coefficients"
                )
            )

        self.x_distortion_coefficients = x_distortion_coefficients
        "X distortion coefficients of the camera"

        if len(y_distortion_coefficients) > self.max_distorsion_coefficients:
            raise ValueError(
                (
                    f"Can't have more than {self.max_distorsion_coefficients} "
                    "distortion coefficients"
                )
            )

        self.y_distortion_coefficients = y_distortion_coefficients
        "Y distortion coefficients of the camera"

        if isinstance(view_port, CameraViewPort):
            view_port = view_port
        elif isinstance(view_port, np.ndarray) and view_port.shape == (
            2,
            2,
        ):
            view_port = CameraViewPort(view_port[0], view_port[1])
        else:
            raise TypeError(
                "view_port must be a CameraViewPort or a (2,2) shape numpy array"
            )

        self.view_port = view_port
        "Camera viewport"

    @staticmethod
    def _build(stream) -> "BTSCameraData":
        rotation_matrix = MAT3X3D.bread(stream)
        translation_vector = VEC3D.bread(stream)
        focus = VEC2D.bread(stream)
        optical_center = VEC2D.bread(stream)
        x_distortion_coefficients = f64.bread(
            stream, BTSCameraData.max_distorsion_coefficients
        )
        y_distortion_coefficients = f64.bread(
            stream, BTSCameraData.max_distorsion_coefficients
        )
        view_port = CameraViewPort.bread(stream)
        return BTSCameraData(
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            focus=focus,
            optical_center=optical_center,
            x_distortion_coefficients=x_distortion_coefficients,
            y_distortion_coefficients=y_distortion_coefficients,
            view_port=view_port,
        )

    def _write(self, file) -> None:
        MAT3X3D.bwrite(file, self.rotation_matrix)  # rotation_matrix
        VEC3D.bwrite(file, self.translation_vector)  # translation_vector
        VEC2D.bwrite(file, self.focus)  # focus
        VEC2D.bwrite(file, self.optical_center)  # optical_center
        f64.bwrite(file, self.x_distortion_coefficients)  # x_distortion_coefficients
        f64.bwrite(file, self.y_distortion_coefficients)  # y_distortion_coefficients
        self.view_port.bwrite(file)  # view_port

    @property
    def nBytes(self) -> int:
        return (
            MAT3X3D.btype.itemsize  # rotation_matrix
            + VEC3D.btype.itemsize  # translation_vector
            + VEC2D.btype.itemsize  # focus
            + VEC2D.btype.itemsize  # optical_center
            + f64.btype.itemsize
            * self.max_distorsion_coefficients  # x_distortion_coefficients
            + f64.btype.itemsize
            * self.max_distorsion_coefficients  # y_distortion_coefficients
            + CameraViewPort.nBytes  # view_port
        )


class CalibrationDataBlockFormat(IntEnum):
    """
    Available block formats for the Calibration Data block
    """

    Unknown = 0
    Seelab1 = 1
    BTS = 2


class CalibrationDataBlock(Block):
    """
    A class representing the Calibration Data block. This block holds
    the calibration data for the cameras, as well as the calibration
    volume data.
    """

    type = BlockType.calibrationData

    def __init__(
        self,
        distorsion_model: DistorsionModel,
        calibration_volume_size: np.ndarray,
        calibration_volume_rotation_matrix: np.ndarray,
        calibration_volume_translation_vector: np.ndarray,
        cameras_calibration_map: np.ndarray,
        cam_data: Union[List[BTSCameraData], List[SeelabCameraData]],
        format: CalibrationDataBlockFormat = CalibrationDataBlockFormat.Seelab1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.distorsion_model = distorsion_model
        "Distorsion model of the calibration"

        if calibration_volume_size.shape != VEC3F.btype.shape:
            raise ValueError(
                (
                    "calibration_volume_size must be a "
                    f"{VEC3F.btype.shape} shape numpy array"
                )
            )

        self.calibration_volume_size = calibration_volume_size
        "Size of the calibration volume"

        if calibration_volume_rotation_matrix.shape != MAT3X3F.btype.shape:
            raise ValueError(
                (
                    "calibration_volume_rotation_matrix must "
                    f"be a {MAT3X3F.btype.shape} shape numpy array"
                )
            )

        self.calibration_volume_rotation_matrix = calibration_volume_rotation_matrix
        "Rotation matrix of the calibration volume"

        if calibration_volume_translation_vector.shape != VEC3F.btype.shape:
            raise ValueError(
                (
                    "calibration_volume_translation_vector must "
                    f"be a {VEC3F.btype.shape} shape numpy array"
                )
            )

        self.calibration_volume_translation_vector = (
            calibration_volume_translation_vector
        )
        "Translation vector of the calibration volume"

        if (
            not isinstance(cameras_calibration_map, np.ndarray)
            or len(cameras_calibration_map.shape) != 1
        ):
            raise ValueError("Cameras_calibration_map must be a single row numpy array")
        self.cameras_calibration_map = cameras_calibration_map

        self.cam_data = cam_data
        self.format = format

    @staticmethod
    def _build(stream, format) -> "CalibrationDataBlock":
        format = CalibrationDataBlockFormat(format)

        nCams = i32.bread(stream)
        distorsion_model = DistorsionModel(i32.bread(stream))
        calibration_volume = VEC3F.bread(stream)
        rotation_matrix = MAT3X3F.bread(stream)
        translation_vector = VEC3F.bread(stream)
        calibration_map = i16.bread(stream, nCams)

        calibration_data = []

        if format == CalibrationDataBlockFormat.Seelab1:
            calibration_data = [SeelabCameraData._build(stream) for _ in range(nCams)]
        elif format == CalibrationDataBlockFormat.BTS:
            calibration_data = [BTSCameraData._build(stream) for _ in range(nCams)]
        else:
            raise ValueError(f'"Unknown calibration format "{format}"')

        return CalibrationDataBlock(
            distorsion_model=distorsion_model,
            calibration_volume_size=calibration_volume,
            calibration_volume_rotation_matrix=rotation_matrix,
            calibration_volume_translation_vector=translation_vector,
            cameras_calibration_map=calibration_map,
            cam_data=calibration_data,
            format=format,
        )

    def _write(self, file) -> None:
        # nCams
        nCams = len(self.cam_data)
        i32.bwrite(file, nCams)

        # DistorsionModel
        i32.bwrite(file, self.distorsion_model)

        # calibration_volume
        VEC3F.bwrite(file, self.calibration_volume_size)

        # rotation matrix
        MAT3X3F.bwrite(file, self.calibration_volume_rotation_matrix)

        # translation_vector
        VEC3F.bwrite(file, self.calibration_volume_translation_vector)

        # calibration map
        i16.bwrite(file, self.cameras_calibration_map)

        # calibration data
        for cam in self.cam_data:
            cam._write(file)

    @property
    def nBytes(self) -> int:
        return (
            i32.btype.itemsize  # nCams
            + i32.btype.itemsize  # DistorsionModel
            + VEC3F.btype.itemsize  # calibration_volume
            + MAT3X3F.btype.itemsize  # rotation matrix
            + VEC3F.btype.itemsize  # translation_vector
            + i16.btype.itemsize * len(self.cam_data)  # calibration map
            + sum(cam.nBytes for cam in self.cam_data)  # calibration data
        )

    def __iter__(self):
        """
        Iterates over the cameras in the block.
        Returns a tuple of (channel, camera)
        """
        return iter(zip(self.cameras_calibration_map, self.cam_data))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, CalibrationDataBlock):
            return False

        return (
            self.distorsion_model == o.distorsion_model
            and np.array_equal(self.calibration_volume_size, o.calibration_volume_size)
            and np.array_equal(
                self.calibration_volume_rotation_matrix,
                o.calibration_volume_rotation_matrix,
            )
            and np.array_equal(
                self.calibration_volume_translation_vector,
                o.calibration_volume_translation_vector,
            )
            and np.array_equal(self.cameras_calibration_map, o.cameras_calibration_map)
            and all(i == j for i, j in zip(self.cam_data, o.cam_data))
            and self.format == o.format
        )

    def __repr__(self) -> str:
        return (
            "<CalibrationDataBlock "
            f"format={self.format.name} "
            f"nCams={len(self.cam_data)} "
            f"distorsion_model={self.distorsion_model.name} "
            f"calibration_volume_size={self.calibration_volume_size} >"
        )
