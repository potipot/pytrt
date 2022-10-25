import numpy as np

from pathlib import Path
from data_processing import ImagePreprocessor

from build_engine import EngineBuilder
from infer import TensorRTInfer
from viz import draw_mask


def main(verbose: bool = False, workspace: int = 2, precision: str = "fp16"):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    path = Path(__file__).parent.parent
    onnx_file_path = path / "models/onnx/uni/multimodel_1664543523784_export_ji_argmax_buslane_v3+lane_layout_v2_448x256_light.onnx"
    engine_file_path = path / f"models/trt/uni/{onnx_file_path.stem}.engine"
    images_dir = path / "images"

    builder = EngineBuilder(verbose, workspace)
    builder.create_network(onnx_file_path)
    builder.create_engine(engine_file_path, precision)

    trt_infer = TensorRTInfer(engine_file_path)

    input_shape_wh = (448, 256)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    processor = ImagePreprocessor(input_shape=input_shape_wh)
    for image_path in images_dir.iterdir():
        image_raw, image_preprocessed = processor.process(image_path)
        results = trt_infer.infer(image_preprocessed)
        mask = results[-1]
        draw_mask(mask, image_path.name)


if __name__ == "__main__":
    main()
