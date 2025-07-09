from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Models(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VGG_FACE: _ClassVar[Models]
    FACENET: _ClassVar[Models]
    FACENET512: _ClassVar[Models]
    OPENFACE: _ClassVar[Models]
    DEEPFACE: _ClassVar[Models]
    DEEPID: _ClassVar[Models]
    ARCFACE: _ClassVar[Models]
    DLIB_MODEL: _ClassVar[Models]
    SFACE: _ClassVar[Models]
    GHOSTFACENET: _ClassVar[Models]
    BUFFALO_L: _ClassVar[Models]

class Detectors(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPENCV: _ClassVar[Detectors]
    SSD: _ClassVar[Detectors]
    DLIB: _ClassVar[Detectors]
    MTCNN: _ClassVar[Detectors]
    FASTMTCNN: _ClassVar[Detectors]
    RETINAFACE: _ClassVar[Detectors]
    MEDIAPIPE: _ClassVar[Detectors]
    YOLOV8: _ClassVar[Detectors]
    YOLOV11S: _ClassVar[Detectors]
    YOLOV11N: _ClassVar[Detectors]
    YOLOV11M: _ClassVar[Detectors]
    YUNET: _ClassVar[Detectors]
    CENTERFACE: _ClassVar[Detectors]

class DistanceMetrics(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COSINE: _ClassVar[DistanceMetrics]
    EUCLIDEAN: _ClassVar[DistanceMetrics]
    EUCLIDEAN_L2: _ClassVar[DistanceMetrics]
    ANGULAR: _ClassVar[DistanceMetrics]
VGG_FACE: Models
FACENET: Models
FACENET512: Models
OPENFACE: Models
DEEPFACE: Models
DEEPID: Models
ARCFACE: Models
DLIB_MODEL: Models
SFACE: Models
GHOSTFACENET: Models
BUFFALO_L: Models
OPENCV: Detectors
SSD: Detectors
DLIB: Detectors
MTCNN: Detectors
FASTMTCNN: Detectors
RETINAFACE: Detectors
MEDIAPIPE: Detectors
YOLOV8: Detectors
YOLOV11S: Detectors
YOLOV11N: Detectors
YOLOV11M: Detectors
YUNET: Detectors
CENTERFACE: Detectors
COSINE: DistanceMetrics
EUCLIDEAN: DistanceMetrics
EUCLIDEAN_L2: DistanceMetrics
ANGULAR: DistanceMetrics

class FacialArea(_message.Message):
    __slots__ = ("left_eye", "right_eye", "mouth_left", "mouth_right", "nose", "h", "w", "x", "y")
    LEFT_EYE_FIELD_NUMBER: _ClassVar[int]
    RIGHT_EYE_FIELD_NUMBER: _ClassVar[int]
    MOUTH_LEFT_FIELD_NUMBER: _ClassVar[int]
    MOUTH_RIGHT_FIELD_NUMBER: _ClassVar[int]
    NOSE_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    left_eye: _containers.RepeatedScalarFieldContainer[int]
    right_eye: _containers.RepeatedScalarFieldContainer[int]
    mouth_left: _containers.RepeatedScalarFieldContainer[int]
    mouth_right: _containers.RepeatedScalarFieldContainer[int]
    nose: _containers.RepeatedScalarFieldContainer[int]
    h: int
    w: int
    x: int
    y: int
    def __init__(self, left_eye: _Optional[_Iterable[int]] = ..., right_eye: _Optional[_Iterable[int]] = ..., mouth_left: _Optional[_Iterable[int]] = ..., mouth_right: _Optional[_Iterable[int]] = ..., nose: _Optional[_Iterable[int]] = ..., h: _Optional[int] = ..., w: _Optional[int] = ..., x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...
