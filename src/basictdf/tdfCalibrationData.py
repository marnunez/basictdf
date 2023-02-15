from enum import IntEnum
from typing import List, Union

import numpy as np

from basictdf.tdfBlock import Block, BlockType
from basictdf.tdfTypes import (
    VEC2D,
    VEC3D,
    VEC3F,
    CameraViewPort,
    i16,
    i32,
    f64,
    MAT3X3D,
    MAT3X3F,
    TdfType,
)

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


class SeelabCameraData:
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
        self.rotation_matrix = rotation_matrix
        "Rotation matrix of the camera"

        self.translation_vector = translation_vector
        "Translation vector of the camera"

        self.focus = focus
        "Focal length of the camera"

        self.optical_center = optical_center
        "Optical center of the camera"

        self.radial_distortion = radial_distortion
        "Radial distortion of the camera"

        self.decentering = decentering
        "Decentering of the camera"

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
                f"view_port must be a CameraViewPort or a (2,2) shape numpy array"
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
                f"Can't have more than {self.max_distorsion_coefficients} distortion coefficients"
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
                f"view_port must be a CameraViewPort or a (2,2) shape numpy array"
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
    type = BlockType.calibrationData

    def __init__(
        self,
        distorsion_model: DistorsionModel,
        calibration_volume_size: np.ndarray,
        calibration_volume_rotation_matrix: np.ndarray,
        calibration_volume_translation_vector: np.ndarray,
        cameras_calibration_map: np.ndarray,
        cam_data: Union[List[BTSCameraData], List[SeelabCameraData]],
    ) -> None:
        super().__init__()

        self.distorsion_model = distorsion_model
        "Distorsion model of the calibration"

        if calibration_volume_size.shape != VEC3F.btype.shape:
            raise ValueError(
                f"calibration_volume_size must be a {VEC3F.btype.shape} shape numpy array"
            )

        self.calibration_volume_size = calibration_volume_size
        "Size of the calibration volume"

        if calibration_volume_rotation_matrix.shape != MAT3X3F.btype.shape:
            raise ValueError(
                f"calibration_volume_rotation_matrix must be a {MAT3X3F.btype.shape} shape numpy array"
            )

        self.calibration_volume_rotation_matrix = calibration_volume_rotation_matrix
        "Rotation matrix of the calibration volume"

        if calibration_volume_translation_vector.shape != VEC3F.btype.shape:
            raise ValueError(
                f"calibration_volume_translation_vector must be a {VEC3F.btype.shape} shape numpy array"
            )

        self.calibration_volume_translation_vector = (
            calibration_volume_translation_vector
        )
        "Translation vector of the calibration volume"

        if (
            not isinstance(cameras_calibration_map, np.ndarray)
            or len(cameras_calibration_map.shape) != 1
        ):
            raise ValueError(
                f"cameras_calibration_map must be a single row numpy array"
            )
        self.cameras_calibration_map = cameras_calibration_map

        self.cam_data = cam_data

    @staticmethod
    def _build(stream, format) -> "CalibrationDataBlock":
        nCams = i32.bread(stream)
        distosion_model = DistorsionModel(i32.bread(stream))
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
            distorsion_model=distosion_model,
            calibration_volume_size=calibration_volume,
            calibration_volume_rotation_matrix=rotation_matrix,
            calibration_volume_translation_vector=translation_vector,
            cameras_calibration_map=calibration_map,
            cam_data=calibration_data,
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

    def nBytes(self) -> int:
        return (
            i32.btype.itemsize  # nCams
            + i32.btype.itemsize  # DistorsionModel
            + VEC3F.btype.itemsize  # calibration_volume
            + MAT3X3F.btype.itemsize  # rotation matrix
            + VEC3F.btype.itemsize  # translation_vector
            + i16.btype.itemsize * len(self.cam_data)  # calibration map
            + sum([cam.nBytes for cam in self.cam_data])  # calibration data
        )