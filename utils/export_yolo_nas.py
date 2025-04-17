from super_gradients.training import models
import super_gradients.training.processing as processing
import numpy as np
import json

def export_onnx(model, model_type, input_shape=(1, 3, 640, 640), opset_version=11):
    model.eval()
    model.prep_model_for_conversion(input_size=input_shape)
    model.export(f"{model_type}.onnx", postprocessing=None, preprocessing=None,
                onnx_export_kwargs={"opset_version":opset_version})
    

def get_preprocessing_steps(preprocessing):
    if isinstance(preprocessing, processing.StandardizeImage):
        return {"StandardizeImage": {"max_value": preprocessing.max_value}}
    elif isinstance(preprocessing, processing.DetectionRescale):
        return {
            "DetectionRescale": {
                "output_shape": preprocessing.output_shape       
            }
        }
    elif isinstance(preprocessing, processing.DetectionLongestMaxSizeRescale):
        return {
            "DetectionLongestMaxSizeRescale": {
              "output_shape": preprocessing.output_shape
            }
        }
    elif isinstance(preprocessing, processing.DetectionBottomRightPadding):
        return {
            "DetectionBottomRightPadding": {
                "pad_value": preprocessing.pad_value,
                "output_shape": preprocessing.output_shape
            }
        }
    elif isinstance(preprocessing, processing.DetectionCenterPadding):
        return {
            "DetectionCenterPadding": {
                "pad_value": preprocessing.pad_value,
                "output_shape": preprocessing.output_shape
            }
        }
    elif isinstance(preprocessing, processing.NormalizeImage):
        return {
            "NormalizeImage": {"mean": preprocessing.mean.tolist(), "std": preprocessing.std.tolist()}
        }
    elif isinstance(preprocessing, processing.ImagePermute):
        return {
            "ImagePermute": {
                "order": preprocessing.permutation
            }
        }
    elif isinstance(preprocessing, processing.ReverseImageChannels):
        return {
            "ReverseImageChannels": True
        }
    else:
        raise NotImplemented("Model have processing steps that haven't been implemented")

def export_metadata(model, model_type):
    preprocessing_steps = [
        get_preprocessing_steps(st) for st in model._image_processor.processings
    ]

    dummy = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)

    input_shape = np.expand_dims(model._image_processor.preprocess_image(dummy)[0], 0).shape

    postprocessing_steps = {
        "NMS": {
            "iou": model._default_nms_iou,
            "conf": model._default_nms_conf
        }
    }

    labels = model.get_class_names()

    res = {
            "type": model_type,
            "input_shape": input_shape,
            "post_processing": postprocessing_steps,
            "pre_processing": preprocessing_steps,
            "labels": labels,
        }

    filename = f"{model_type}-metadata.json"
    with open(filename, "w") as f:
        f.write(json.dumps(res))

if __name__ == "__main__":
    model_type = "yolo_nas_s"
    model = models.get(model_type, pretrained_weights="coco")
    export_metadata(model, model_type)
    export_onnx(model, model_type)