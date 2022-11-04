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

import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class _TRTInferenceBase:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path, logger: trt.Logger = None):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        if logger is None:
            logger = trt.Logger(trt.Logger.INFO)
        self.logger = logger
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        # Load TRT engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.setup_io_bindings()

    def setup_io_bindings(self):
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        return [(o["shape"], o["dtype"]) for o in self.outputs]


class UniModelInference(_TRTInferenceBase):
    def infer(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Prepare the output data
        outputs = [np.zeros(shape, dtype) for shape, dtype in self.output_spec()]
        # Process I/O and execute the network
        inference_start_time = time.time()
        for mem_placeholder, model_input in zip(self.inputs, batch):
            cuda.memcpy_htod(mem_placeholder["allocation"], np.ascontiguousarray(model_input))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]["allocation"])
        inference_time = round((time.time() - inference_start_time) * 1000, ndigits=3)
        message = f"inference time: {inference_time} ms"
        self.logger.log(trt.Logger.Severity.INFO, message)
        return outputs


class SplitModelInference:
    def __init__(self, encoder_engine_path, decoders_engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.encoder = _TRTInferenceBase(encoder_engine_path, logger=self.logger)
        self.decoders = _TRTInferenceBase(decoders_engine_path, logger=self.logger)

    def infer(self, batch):
        # Prepare the output data
        outputs = [np.zeros(shape, dtype) for shape, dtype in self.decoders.output_spec()]
        inference_start_time = time.time()

        # copy data from python to cuda
        for mem_placeholder, model_input in zip(self.encoder.inputs, batch):
            cuda.memcpy_htod(mem_placeholder["allocation"], np.ascontiguousarray(model_input))

        # execute encoder context
        self.encoder.context.execute_v2(self.encoder.allocations)
        encoder_inference_time = round((time.time() - inference_start_time) * 1000, ndigits=3)

        decoders_start_time = time.time()
        # TODO: avoid copying in favor of sharing space?
        # copy data from encoders output to decoders input
        for decoders_input, encoder_output in zip(self.decoders.inputs, self.encoder.outputs):
            memory_size = np.zeros(decoders_input["shape"], decoders_input["dtype"]).nbytes
            cuda.memcpy_dtod(decoders_input["allocation"], encoder_output["allocation"], memory_size)

        # execute decoders context
        self.decoders.context.execute_v2(self.decoders.allocations)

        # copy data from cuda to python
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.decoders.outputs[o]["allocation"])

        decoders_inference_time = round((time.time() - decoders_start_time) * 1000, ndigits=3)
        total_inference_time = round(encoder_inference_time + decoders_inference_time, ndigits=3)
        message = f"inference time: {total_inference_time} ms (encoder: {encoder_inference_time}, decoders: {decoders_inference_time})"
        self.logger.log(trt.Logger.Severity.INFO, message)
        return outputs
