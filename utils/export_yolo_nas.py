import argparse
import json
import os

import numpy as np
import super_gradients.training.processing as processing
from super_gradients.conversion import ExportQuantizationMode
from super_gradients.training import models


def export_onnx(model, model_type, output_path=None, input_shape=(1, 3, 640, 640), opset_version=11, quantization=None):
    quantization_mode = ExportQuantizationMode(quantization) if quantization else None
    model_name = model_type if not quantization else f"{model_type}_{quantization}"

    out_file = f"{model_name}.onnx"
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, out_file)

    model.eval()
    model.prep_model_for_conversion(input_size=input_shape)
    model.export(out_file, postprocessing=None, preprocessing=None, quantization_mode=quantization_mode,
                onnx_export_kwargs={"opset_version":opset_version})
    return out_file

def export_openvino(onnx_file):
    import openvino as ov
    model_path = onnx_file.remove_suffix('.onnx')
    model = ov.convert_model(onnx_file)
    ov.serialize(model, f"{model_path}.xml", f"{model_path}.bin")

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

def export_metadata_get_input_shape(model, model_type, output_path=None) -> tuple:
    out_file = f"{model_type}-metadata.json"
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, out_file)

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

    metadata = {
            "type": model_type,
            "input_shape": input_shape,
            "post_processing": postprocessing_steps,
            "pre_processing": preprocessing_steps,
            "labels": labels,
        }

    with open(out_file, "w") as f:
        f.write(json.dumps(metadata))
    return input_shape

def main(model_type, opset_version, output_path=None, openvino=False, quantization=None):
    model = models.get(model_type, pretrained_weights="coco")
    input_shape = export_metadata_get_input_shape(model, model_type, output_path=output_path)
    onnx_file = export_onnx(model, model_type, output_path=output_path,  input_shape=input_shape, opset_version=opset_version, quantization=quantization)
    if openvino:
        export_openvino(onnx_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"],
                        help="Model type (small, medium, large)")
    parser.add_argument("--opset_version", type=int, default=11,
                        help="ONNX opset version (default=11), "
                        "this is required in order to use onnxruntime compiled with OpenCV 4.6.0, "
                        "not the same as onnxruntime version")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory to export files, "
                        "will be exported to CWD if not specified")
    parser.add_argument("--openvino", action='store_true', help="Export OpenVINO model")
    parser.add_argument("--quantization", type=str, required=False, choices=["int8", "fp16"], default=None,
                        help="Quantize model to int8, fp16(requires GPU), or leave unspecified for no quantization")
    args = parser.parse_args()

    main(args.model_type, args.opset_version, output_path=args.output_path, openvino=args.openvino, quantization=args.quantization)
