from pathlib import Path
from data_processing import ImagePreprocessor
from loguru import logger

from build_engine import EngineBuilder
from infer import TensorRTInfer
from viz import draw_mask


def main(verbose: bool = False, workspace: int = 2, precision: str = "fp16"):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    path = Path(__file__).parent.parent.resolve()
    encoder_onnx_filepath = path / "models/onnx/dual/encoder.onnx"
    decoders_onnx_filepath = path / "models/onnx/dual/decoders.onnx"
    encoder_engine_filepath = path / f"models/trt/dual/{encoder_onnx_filepath.stem}.engine"
    decoders_engine_filepath = path / f"models/trt/dual/{decoders_onnx_filepath.stem}.engine"

    images_dir = path / "images"

    builder = EngineBuilder(verbose, workspace)

    builder.create_network(encoder_onnx_filepath)
    builder.create_engine(encoder_engine_filepath, precision)
    builder.create_network(decoders_onnx_filepath)
    builder.create_engine(decoders_engine_filepath, precision)

    encoder_inference = TensorRTInfer(encoder_engine_filepath)
    decoders_inference = TensorRTInfer(decoders_engine_filepath)

    input_shape_wh = (448, 256)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    processor = ImagePreprocessor(input_shape=input_shape_wh)
    for image_path in images_dir.iterdir():
        image_raw, image_preprocessed = processor.process(image_path)
        encoder_output = encoder_inference.infer(image_preprocessed)
        logger.debug(f"Encoder inference time: {encoder_inference.inference_time} ms")
        feature_map, *_ = encoder_output
        decoders_output = decoders_inference.infer(feature_map)
        logger.debug(f"Decoders inference time: {decoders_inference.inference_time} ms")
        logger.info(f"Total inference time: {encoder_inference.inference_time+decoders_inference.inference_time} ms")
        mask = decoders_output[-1]
        draw_mask(mask, image_path.name)


if __name__ == "__main__":
    main(verbose=True)
