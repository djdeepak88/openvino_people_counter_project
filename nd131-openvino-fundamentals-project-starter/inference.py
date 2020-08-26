#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging

#Create and configure logger
logging.basicConfig(filename="people_counter.log",format='%(asctime)s %(message)s',filemode='a')

#Creating an object
log=logging.getLogger()

#Set debug logging level
log.setLevel(logging.DEBUG)

from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.net = None
        ie = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None


    def load_model(self, model, device):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        log.info("Reading the model xml and binary")
        plugin = IECore()
        log.info("Loading the inference engine.")
        ### Load IR files into their related class
        self.net = IENetwork(model=model_xml, weights=model_bin)
        ### TODO: Check for supported layers ###
        supported_layers = plugin.query_network(self.net, device)

        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        ### TODO: Add any necessary extensions ###
        # No extension required for 2020.4 version of Openvino toolkit.
        ### TODO: Return the loaded inference plugin ###
        ### Load the network into the Inference Engine
        self.net_plugin = plugin.load_network(network = self.net,  device_name=device)

        log.info("Network input structure")
        log.info(list(self.net.inputs))
        log.info(list(self.net.inputs)[0])
        log.info(list(self.net.inputs)[1])
        log.info(self.net.inputs['image_info'].shape)
        log.info(self.net.inputs['image_tensor'].shape)
        self.input_blob = list(self.net.inputs)[1]
        log.info("Input blob")
        log.info(self.input_blob)
        log.info("Network output structure")
        log.info(list(self.net.outputs))
        log.info(self.net.outputs['detection_output'].shape)
        self.out_blob = list(self.net.outputs)[0]
        log.info(self.out_blob)
        return plugin, self.get_input_shape()

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        log.info("Input blob dimension:-")
        log.info(self.net.inputs[self.input_blob].shape)
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, frame):
        ### TODO: Start an asynchronous request ###
        self.infer_request_handle = self.net_plugin.start_async(request_id=0, inputs={self.input_blob: frame})
        log.info("Inference Request Handle")
        log.info(self.infer_request_handle)
        return self.net_plugin

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        infer_status = self.net_plugin.requests[0].wait(-1)
        log.info("Inference status")
        log.info(infer_status)
        return infer_status

    def get_output(self, output=None):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###

        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net_plugin.requests[0].outputs[self.out_blob]

        log.info("Result")
        log.info(res)

        return res

    def clean(self):
        del self.net_plugin
        del self.net
