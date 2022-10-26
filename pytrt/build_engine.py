#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
from pathlib import Path

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from loguru import logger


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose: bool = False, workspace: int = 8, use_dla: bool = False, precision: str = "fp16"):
        """
        @param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        @param use_dla: Uses DLA (deep learning accelerator) if available
        @param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        @param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = workspace * (2 ** 30)

        if use_dla:
            self.config.default_device_type = trt.DeviceType.DLA
            self.config.DLA_core = 0

        if precision == "fp16":
            if self.builder.platform_has_fast_fp16:
                self.config.set_flag(trt.BuilderFlag.FP16)
            else:
                logger.warning("FP16 was selected but is not supported natively on this platform/device")

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path: Path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                logger.error(f"Failed to load ONNX file: {onnx_path}")
                for error in range(self.parser.num_errors):
                    logger.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        logger.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            logger.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")

        for output in outputs:
            logger.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")

        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size

    def create_engine(self, engine_path: Path, overwrite: bool = False):
        """
        Build the TensorRT engine and serialize it to disk.
        @param engine_path: The path where to serialize the engine to.
        @param overwrite: whether to overwrite an existing TRT engine file
        """

        if engine_path.exists() and not overwrite:
            logger.info(f"Engine filepath already exists and overwrite is set to False: {engine_path}")
        else:
            logger.info(f"Building Engine in {engine_path}")
            with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
                logger.info(f"Serializing engine to file: {engine_path}")
                f.write(engine.serialize())
