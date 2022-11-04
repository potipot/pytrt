from pathlib import Path
from data_processing import ImagePreprocessor
import fire

from build_engine import EngineBuilder
from infer import UniModelInference, SplitModelInference
from viz import draw_mask


def main(verbose: bool = False, workspace: int = 2, precision: str = "fp16", use_dla: bool = False):
    """Create a TensorRT engine for ONNX-based multimodel and run inference."""

    path = Path(__file__).parent.parent.resolve()
    encoder_onnx_filepath = path / "models/onnx/dual/linknet_encoder.onnx"
    decoders_onnx_filepath = path / "models/onnx/dual/linknet_decoders.onnx"
    encoder_engine_filepath = (
        path / f"models/trt/dual/{encoder_onnx_filepath.stem}_{precision}_{'dla' if use_dla else 'gpu'}.engine"
    )
    decoders_engine_filepath = path / f"models/trt/dual/{decoders_onnx_filepath.stem}_{precision}.engine"

    encoder_builder = EngineBuilder(verbose, workspace, use_dla=use_dla, precision=precision)
    encoder_builder.create_network(encoder_onnx_filepath)
    encoder_builder.create_engine(encoder_engine_filepath)

    decoder_builder = EngineBuilder(verbose, workspace, precision=precision)
    decoder_builder.create_network(decoders_onnx_filepath)
    decoder_builder.create_engine(decoders_engine_filepath)

    inference = SplitModelInference(encoder_engine_filepath, decoders_engine_filepath)
    # inference = UniModelInference(path_to_unimodel)
    input_shape_wh = (448, 256)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    images_dir = path / "images"
    processor = ImagePreprocessor(input_shape=input_shape_wh)
    for image_path in images_dir.iterdir():
        image_raw, image_preprocessed = processor.process(image_path)
        output = inference.infer(image_preprocessed)
        mask = output[-1]
        draw_mask(mask, image_path.name)


if __name__ == "__main__":
    fire.Fire(main)
