import grpc
from commons import image_utils
from deepface.commons.logger import Logger
from deepface.api.proto.common_pb2 import FacialArea
import deepface.api.proto.analyze_pb2_grpc as analyze_grpc
import deepface.api.proto.analyze_pb2 as analyze
import deepface.api.proto.represent_pb2_grpc as represent_grpc
import deepface.api.proto.represent_pb2 as represent
import deepface.api.proto.verify_pb2_grpc as verify_grpc
import deepface.api.proto.verify_pb2 as verify

from deepface import DeepFace

from deepface.api.src.modules.core.common import (
    default_detector_backend,
    default_enforce_detection,
    default_align,
    default_anti_spoofing,
    default_max_faces,
    default_model_name,
    default_distance_metric,
)

logger = Logger()

class AnalyzeService(analyze_grpc.AnalyzeServiceServicer):

    def Analyze(self, request, context) -> analyze.AnalyzeResponse:
        response = analyze.AnalyzeResponse()

        try:
            demographies = DeepFace.analyze(
                img_path=image_utils.load_image_from_io_object(request.image),
                actions=actions_enum_to_string(request.actions),
                enforce_detection=request.enforce_detection
                if request.HasField("enforce_detection") else
                default_enforce_detection,
                detector_backend=request.detector_backend
                if request.HasField("detector_backend") else
                default_detector_backend,
                align=request.align
                if request.HasField("align") else default_align,
                anti_spoofing=request.anti_spoofing if
                request.HasField("anti_spoofing") else default_anti_spoofing,
            )
        except Exception as err:
            context.set_details(f"Exception while analyzing: {str(err)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return response

        for demography in demographies:
            if isinstance(demography, list):
                demography = demography[0]
            result = response.results.add()
            if "age" in demography:
                result.age = int(demography["age"])
            if "gender" in demography:
                result.gender = demography["gender"]
            if "face_confidence" in demography:
                result.face_confidence = float(demography["face_confidence"])
            if "dominant_emotion" in demography:
                result.dominant_emotion = demography["dominant_emotion"]
            if "dominant_gender" in demography:
                result.dominant_gender = demography["dominant_gender"]
            if "dominant_race" in demography:
                result.dominant_race = demography["dominant_race"]
            if "race" in demography:
                if "asian" in demography["race"]:
                    result.race.asian = demography["race"]["asian"]
                if "indian" in demography["race"]:
                    result.race.indian = demography["race"]["indian"]
                if "black" in demography["race"]:
                    result.race.black = demography["race"]["black"]
                if "white" in demography["race"]:
                    result.race.white = demography["race"]["white"]
                if "middle eastern" in demography["race"]:
                    result.race.middle_eastern = demography["race"]["middle eastern"]
                if "latino hispanic" in demography["race"]:
                    result.race.latino_hispanic = demography["race"]["latino hispanic"]
            if "region" in demography:
                result.facial_area = get_facial_area(demography["region"])
            if "emotion" in demography:
                if "angry" in demography["emotion"]:
                    result.emotion.angry = demography["emotion"]["angry"]
                if "disgust" in demography["emotion"]:
                    result.emotion.disgust = demography["emotion"]["disgust"]
                if "fear" in demography["emotion"]:
                    result.emotion.fear = demography["emotion"]["fear"]
                if "happy" in demography["emotion"]:
                    result.emotion.happy = demography["emotion"]["happy"]
                if "sad" in demography["emotion"]:
                    result.emotion.sad = demography["emotion"]["sad"]
                if "surprise" in demography["emotion"]:
                    result.emotion.surprise = demography["emotion"]["surprise"]
                if "neutral" in demography["emotion"]:
                    result.emotion.neutral = demography["emotion"]["neutral"]

        logger.debug(demographies)

        return response


class RepresentService(represent_grpc.RepresentServiceServicer):

    def Represent(self, request, context) -> represent.RepresentResponse:
        response = represent.RepresentResponse()

        try:
            results = DeepFace.represent(
                img_path=image_utils.load_image_from_io_object(request.image),
                model_name=request.model_name
                if request.HasField("model_name") else default_model_name,
                detector_backend=request.detector_backend
                if request.HasField("detector_backend") else
                default_detector_backend,
                enforce_detection=request.enforce_detection
                if request.HasField("enforce_detection") else
                default_enforce_detection,
                align=request.align
                if request.HasField("align") else default_align,
                anti_spoofing=request.anti_spoofing if
                request.HasField("anti_spoofing") else default_anti_spoofing,
                max_faces=request.max_faces
                if request.HasField("max_faces") else default_max_faces,
            )
        except Exception as err:
            context.set_details(f"Exception while representing: {str(err)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return response

        for result in results:
            if isinstance(result, list):
                result = result[0]
                rep = response.results.add()
                if "embedding" in result:
                    rep.embedding.extend(result["embedding"])
                if "face_confidence" in result:
                    rep.face_confidence = float(result["face_confidence"])
                if "facial_area" in result:
                    rep.facial_area = get_facial_area(result["facial_area"])

        logger.debug(results)

        return response


class VerifyService(verify_grpc.VerifyServiceServicer):

    def Verify(self, request, context) -> verify.VerifyResponse:
        results = DeepFace.verify(
            img1_path=image_utils.load_image_from_io_object(request.img1),
            img2_path=image_utils.load_image_from_io_object(request.img2),
            model_name=request.model_name
            if request.HasField("model_name") else default_model_name,
            detector_backend=request.detector_backend if
            request.HasField("detector_backend") else default_detector_backend,
            distance_metric=request.distance_metric if
            request.HasField("distance_metric") else default_distance_metric,
            align=request.align
            if request.HasField("align") else default_align,
            enforce_detection=request.enforce_detection
            if request.HasField("enforce_detection") else
            default_enforce_detection,
            anti_spoofing=request.anti_spoofing
            if request.HasField("anti_spoofing") else default_anti_spoofing,
        )
        response = verify.VerifyResponse()
        if "verified" in results:
            response.verified = bool(results["verified"])
        if "distance" in results:
            response.distance = float(results["distance"])
        if "facial_areas" in results:
            facial_areas = results["facial_areas"]
            response.facial_areas.img1 = get_facial_area(facial_areas["img1"])
            response.facial_areas.img2 = get_facial_area(facial_areas["img2"])
        if "threshold" in results:
            response.threshold = float(results["threshold"])
        if "time" in results:
            response.time = float(results["time"])
        if "similarity_metric" in results:
            response.similarity_metric = results["similarity_metric"]
        if "detector_backend" in results:
            response.detector_backend = results["detector_backend"]
        if "model" in results:
            response.model = results["model"]

        logger.debug(results)

        return response


def actions_enum_to_string(actions) -> list[str]:
    """
    Convert the actions enum to a list of action names.
    """
    action_names = []
    for action in actions:
        match action:
            case analyze.AnalyzeRequest.Action.AGE:
                action_names.append("age")
            case analyze.AnalyzeRequest.Action.GENDER:
                action_names.append("gender")
            case analyze.AnalyzeRequest.Action.RACE:
                action_names.append("race")
            case analyze.AnalyzeRequest.Action.EMOTION:
                action_names.append("emotion")
    return action_names


def get_facial_area(dict) -> FacialArea:
    """
    Extract the facial area from the dict.
    """
    result = FacialArea()
    result.left_eye = dict.get("left_eye", [0])
    result.right_eye = dict.get("right_eye", [0])
    result.mouth_left = dict.get("mouth_left", [0])
    result.mouth_right = dict.get("mouth_right", [0])
    result.nose = dict.get("nose", [0])
    result.h = int(dict.get("h", 0))
    result.w = int(dict.get("w", 0))
    result.x = int(dict.get("x", 0))
    result.y = int(dict.get("y", 0))
    return result
