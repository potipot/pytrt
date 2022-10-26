from pathlib import Path
from data_processing import ImagePreprocessor
from loguru import logger
import fire

from build_engine import EngineBuilder
from infer import TensorRTInfer
from viz import draw_mask


def main(verbose: bool = False, workspace: int = 2, precision: str = "fp16", use_dla: bool = False):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    path = Path(__file__).parent.parent.resolve()
    encoder_onnx_filepath = path / "models/onnx/dual/encoder.onnx"
    decoders_onnx_filepath = path / "models/onnx/dual/decoders.onnx"
    encoder_engine_filepath = path / f"models/trt/dual/{encoder_onnx_filepath.stem}_{'dla' if use_dla else 'gpu'}.engine"
    decoders_engine_filepath = path / f"models/trt/dual/{decoders_onnx_filepath.stem}.engine"

    images_dir = path / "images"

    encoder_builder = EngineBuilder(verbose, workspace, use_dla=use_dla)
    encoder_builder.create_network(encoder_onnx_filepath)
    encoder_builder.create_engine(encoder_engine_filepath, precision)

    decoder_builder = EngineBuilder(verbose, workspace)
    decoder_builder.create_network(decoders_onnx_filepath)
    decoder_builder.create_engine(decoders_engine_filepath, precision)

    encoder_inference = TensorRTInfer(encoder_engine_filepath)
    decoders_inference = TensorRTInfer(decoders_engine_filepath)

    input_shape_wh = (448, 256)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    processor = ImagePreprocessor(input_shape=input_shape_wh)
    for image_path in images_dir.iterdir():
        image_raw, image_preprocessed = processor.process(image_path)
        encoder_output = encoder_inference.infer(image_preprocessed)
        feature_map, *_ = encoder_output
        decoders_output = decoders_inference.infer(feature_map)
        encoder_inference_time = encoder_inference.inference_time
        decoders_inference_time = decoders_inference.inference_time
        total_inference_time = round(encoder_inference.inference_time+decoders_inference.inference_time, ndigits=3)
        logger.info(f"inference time: {total_inference_time} ms (encoder: {encoder_inference_time}, decoders: {decoders_inference_time}")
        mask = decoders_output[-1]
        draw_mask(mask, image_path.name)


if __name__ == "__main__":
    fire.Fire(main)
