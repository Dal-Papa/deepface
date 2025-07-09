import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RepresentRequest(_message.Message):
    __slots__ = ("image", "model_name", "detector_backend", "enforce_detection", "align", "anti_spoofing", "max_faces")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    ENFORCE_DETECTION_FIELD_NUMBER: _ClassVar[int]
    ALIGN_FIELD_NUMBER: _ClassVar[int]
    ANTI_SPOOFING_FIELD_NUMBER: _ClassVar[int]
    MAX_FACES_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    model_name: _common_pb2.Models
    detector_backend: _common_pb2.Detectors
    enforce_detection: bool
    align: bool
    anti_spoofing: bool
    max_faces: int
    def __init__(self, image: _Optional[bytes] = ..., model_name: _Optional[_Union[_common_pb2.Models, str]] = ..., detector_backend: _Optional[_Union[_common_pb2.Detectors, str]] = ..., enforce_detection: bool = ..., align: bool = ..., anti_spoofing: bool = ..., max_faces: _Optional[int] = ...) -> None: ...

class RepresentResponse(_message.Message):
    __slots__ = ("results",)
    class Results(_message.Message):
        __slots__ = ("embedding", "face_confidence", "facial_area")
        EMBEDDING_FIELD_NUMBER: _ClassVar[int]
        FACE_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
        FACIAL_AREA_FIELD_NUMBER: _ClassVar[int]
        embedding: _containers.RepeatedScalarFieldContainer[float]
        face_confidence: float
        facial_area: _common_pb2.FacialArea
        def __init__(self, embedding: _Optional[_Iterable[float]] = ..., face_confidence: _Optional[float] = ..., facial_area: _Optional[_Union[_common_pb2.FacialArea, _Mapping]] = ...) -> None: ...
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[RepresentResponse.Results]
    def __init__(self, results: _Optional[_Iterable[_Union[RepresentResponse.Results, _Mapping]]] = ...) -> None: ...
