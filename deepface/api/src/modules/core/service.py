# built-in dependencies
import traceback
from typing import Optional, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=broad-except


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        faces = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = [f.to_dict() for f in faces]
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    anti_spoofing: bool,
):
    try:
        verif = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            anti_spoofing=anti_spoofing,
        )
        return verif.to_dict()
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: Union[str, np.ndarray],
    actions: list,
    detector_backend: str,
    anti_spoofing: bool,
):
    try:
        result = {}
        faces = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = [f.to_dict() for f in faces]
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400
