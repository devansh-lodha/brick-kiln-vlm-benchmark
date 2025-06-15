# src/inference/tiled_inference.py

from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
from typing import Any, List, Dict
import numpy as np

def get_sahi_model_for_vlm(model: Any, processor: Any, prompt: str, class_names: List[str]):
    """
    Creates a SAHI-compatible model object from a loaded VLM.
    This is a conceptual wrapper to make any VLM work with SAHI's pipeline.
    """
    from sahi.models.base import DetectionModel
    from sahi.prediction import ObjectPrediction
    from .predict import predict_vlm
    from ..evaluation.metrics import parse_vlm_output_to_boxes

    class VLMDetectionModel(DetectionModel):
        def load_model(self):
            self.model = model
            self.processor = processor
            self.prompt = prompt
            self.category_mapping = {i: name for i, name in enumerate(class_names)}

        def perform_inference(self, image: np.ndarray):
            pil_image = Image.fromarray(image)
            raw_output = predict_vlm(self.model, self.processor, pil_image, self.prompt)
            
            # Use the parsing logic from our metrics module
            parsed_results = parse_vlm_output_to_boxes(raw_output, class_names)
            
            # Convert to SAHI's ObjectPrediction format
            self._object_prediction_list = []
            for res in parsed_results:
                self._object_prediction_list.append(
                    ObjectPrediction(
                        bbox=res['box'],
                        category_id=res['class_id'],
                        score=res.get('confidence', 0.99), # Assume high confidence
                        category_name=self.category_mapping[res['class_id']]
                    )
                )

        @property
        def object_prediction_list(self):
            return self._object_prediction_list

    sahi_model = VLMDetectionModel(model_path=None) # model_path is not used
    sahi_model.load_model()
    return sahi_model


def run_tiled_prediction(
    image_path: str,
    sahi_model: Any,
    slice_height: int = 320,
    slice_width: int = 320,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2
) -> List[Dict]:
    """
    Performs tiled inference on a single large image using SAHI.

    Args:
        image_path (str): Path to the large image file.
        sahi_model (Any): The SAHI-compatible model object.
        slice_height (int): Height of each slice.
        slice_width (int): Width of each slice.
        overlap_height_ratio (float): Overlap ratio between vertical slices.
        overlap_width_ratio (float): Overlap ratio between horizontal slices.

    Returns:
        List[Dict]: A list of detected objects in the format SAHI provides.
    """
    
    result = get_sliced_prediction(
        image_path,
        sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.5,
    )
    
    return result.to_coco_annotations()