import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VerifyRequest(_message.Message):
    __slots__ = ("image1", "image2", "model_name", "detector_backend", "distance_metric", "enforce_detection", "align", "anti_spoofing")
    IMAGE1_FIELD_NUMBER: _ClassVar[int]
    IMAGE2_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METRIC_FIELD_NUMBER: _ClassVar[int]
    ENFORCE_DETECTION_FIELD_NUMBER: _ClassVar[int]
    ALIGN_FIELD_NUMBER: _ClassVar[int]
    ANTI_SPOOFING_FIELD_NUMBER: _ClassVar[int]
    image1: bytes
    image2: bytes
    model_name: _common_pb2.Models
    detector_backend: _common_pb2.Detectors
    distance_metric: _common_pb2.DistanceMetrics
    enforce_detection: bool
    align: bool
    anti_spoofing: bool
    def __init__(self, image1: _Optional[bytes] = ..., image2: _Optional[bytes] = ..., model_name: _Optional[_Union[_common_pb2.Models, str]] = ..., detector_backend: _Optional[_Union[_common_pb2.Detectors, str]] = ..., distance_metric: _Optional[_Union[_common_pb2.DistanceMetrics, str]] = ..., enforce_detection: bool = ..., align: bool = ..., anti_spoofing: bool = ...) -> None: ...

class VerifyResponse(_message.Message):
    __slots__ = ("verified", "detector_backend", "model", "similarity_metric", "facial_areas", "distance", "threshold", "time")
    class FacialAreas(_message.Message):
        __slots__ = ("img1", "img2")
        IMG1_FIELD_NUMBER: _ClassVar[int]
        IMG2_FIELD_NUMBER: _ClassVar[int]
        img1: _common_pb2.FacialArea
        img2: _common_pb2.FacialArea
        def __init__(self, img1: _Optional[_Union[_common_pb2.FacialArea, _Mapping]] = ..., img2: _Optional[_Union[_common_pb2.FacialArea, _Mapping]] = ...) -> None: ...
    VERIFIED_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_METRIC_FIELD_NUMBER: _ClassVar[int]
    FACIAL_AREAS_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    verified: bool
    detector_backend: _common_pb2.Detectors
    model: _common_pb2.Models
    similarity_metric: _common_pb2.DistanceMetrics
    facial_areas: VerifyResponse.FacialAreas
    distance: float
    threshold: float
    time: float
    def __init__(self, verified: bool = ..., detector_backend: _Optional[_Union[_common_pb2.Detectors, str]] = ..., model: _Optional[_Union[_common_pb2.Models, str]] = ..., similarity_metric: _Optional[_Union[_common_pb2.DistanceMetrics, str]] = ..., facial_areas: _Optional[_Union[VerifyResponse.FacialAreas, _Mapping]] = ..., distance: _Optional[float] = ..., threshold: _Optional[float] = ..., time: _Optional[float] = ...) -> None: ...
