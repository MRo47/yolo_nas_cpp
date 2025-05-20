import openvino as ov
import argparse

def export_openvino(onnx_file):
    model_path = onnx_file.remove_suffix('.onnx')
    model = ov.convert_model(onnx_file)
    ov.serialize(model, f"{model_path}.xml", f"{model_path}.bin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True, help="Path to the ONNX model file")
    args = parser.parse_args()
    export_openvino(args.model_path)