import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AnalyzeRequest(_message.Message):
    __slots__ = ("image", "actions", "detector_backend", "enforce_detection", "align", "anti_spoofing")
    class Action(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        AGE: _ClassVar[AnalyzeRequest.Action]
        GENDER: _ClassVar[AnalyzeRequest.Action]
        EMOTION: _ClassVar[AnalyzeRequest.Action]
        RACE: _ClassVar[AnalyzeRequest.Action]
    AGE: AnalyzeRequest.Action
    GENDER: AnalyzeRequest.Action
    EMOTION: AnalyzeRequest.Action
    RACE: AnalyzeRequest.Action
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    ENFORCE_DETECTION_FIELD_NUMBER: _ClassVar[int]
    ALIGN_FIELD_NUMBER: _ClassVar[int]
    ANTI_SPOOFING_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    actions: _containers.RepeatedScalarFieldContainer[AnalyzeRequest.Action]
    detector_backend: _common_pb2.Detectors
    enforce_detection: bool
    align: bool
    anti_spoofing: bool
    def __init__(self, image: _Optional[bytes] = ..., actions: _Optional[_Iterable[_Union[AnalyzeRequest.Action, str]]] = ..., detector_backend: _Optional[_Union[_common_pb2.Detectors, str]] = ..., enforce_detection: bool = ..., align: bool = ..., anti_spoofing: bool = ...) -> None: ...

class AnalyzeResponse(_message.Message):
    __slots__ = ("results",)
    class Emotion(_message.Message):
        __slots__ = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
        ANGRY_FIELD_NUMBER: _ClassVar[int]
        DISGUST_FIELD_NUMBER: _ClassVar[int]
        FEAR_FIELD_NUMBER: _ClassVar[int]
        HAPPY_FIELD_NUMBER: _ClassVar[int]
        NEUTRAL_FIELD_NUMBER: _ClassVar[int]
        SAD_FIELD_NUMBER: _ClassVar[int]
        SURPRISE_FIELD_NUMBER: _ClassVar[int]
        angry: float
        disgust: float
        fear: float
        happy: float
        neutral: float
        sad: float
        surprise: float
        def __init__(self, angry: _Optional[float] = ..., disgust: _Optional[float] = ..., fear: _Optional[float] = ..., happy: _Optional[float] = ..., neutral: _Optional[float] = ..., sad: _Optional[float] = ..., surprise: _Optional[float] = ...) -> None: ...
    class Gender(_message.Message):
        __slots__ = ("man", "woman")
        MAN_FIELD_NUMBER: _ClassVar[int]
        WOMAN_FIELD_NUMBER: _ClassVar[int]
        man: float
        woman: float
        def __init__(self, man: _Optional[float] = ..., woman: _Optional[float] = ...) -> None: ...
    class Race(_message.Message):
        __slots__ = ("asian", "black", "indian", "latino_hispanic", "middle_eastern", "white")
        ASIAN_FIELD_NUMBER: _ClassVar[int]
        BLACK_FIELD_NUMBER: _ClassVar[int]
        INDIAN_FIELD_NUMBER: _ClassVar[int]
        LATINO_HISPANIC_FIELD_NUMBER: _ClassVar[int]
        MIDDLE_EASTERN_FIELD_NUMBER: _ClassVar[int]
        WHITE_FIELD_NUMBER: _ClassVar[int]
        asian: float
        black: float
        indian: float
        latino_hispanic: float
        middle_eastern: float
        white: float
        def __init__(self, asian: _Optional[float] = ..., black: _Optional[float] = ..., indian: _Optional[float] = ..., latino_hispanic: _Optional[float] = ..., middle_eastern: _Optional[float] = ..., white: _Optional[float] = ...) -> None: ...
    class Result(_message.Message):
        __slots__ = ("age", "dominant_emotion", "dominant_gender", "dominant_race", "face_confidence", "emotion", "gender", "race", "facial_area")
        AGE_FIELD_NUMBER: _ClassVar[int]
        DOMINANT_EMOTION_FIELD_NUMBER: _ClassVar[int]
        DOMINANT_GENDER_FIELD_NUMBER: _ClassVar[int]
        DOMINANT_RACE_FIELD_NUMBER: _ClassVar[int]
        FACE_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
        EMOTION_FIELD_NUMBER: _ClassVar[int]
        GENDER_FIELD_NUMBER: _ClassVar[int]
        RACE_FIELD_NUMBER: _ClassVar[int]
        FACIAL_AREA_FIELD_NUMBER: _ClassVar[int]
        age: int
        dominant_emotion: str
        dominant_gender: str
        dominant_race: str
        face_confidence: float
        emotion: AnalyzeResponse.Emotion
        gender: AnalyzeResponse.Gender
        race: AnalyzeResponse.Race
        facial_area: _common_pb2.FacialArea
        def __init__(self, age: _Optional[int] = ..., dominant_emotion: _Optional[str] = ..., dominant_gender: _Optional[str] = ..., dominant_race: _Optional[str] = ..., face_confidence: _Optional[float] = ..., emotion: _Optional[_Union[AnalyzeResponse.Emotion, _Mapping]] = ..., gender: _Optional[_Union[AnalyzeResponse.Gender, _Mapping]] = ..., race: _Optional[_Union[AnalyzeResponse.Race, _Mapping]] = ..., facial_area: _Optional[_Union[_common_pb2.FacialArea, _Mapping]] = ...) -> None: ...
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[AnalyzeResponse.Result]
    def __init__(self, results: _Optional[_Iterable[_Union[AnalyzeResponse.Result, _Mapping]]] = ...) -> None: ...
