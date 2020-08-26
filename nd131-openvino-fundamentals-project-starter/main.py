"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2
import math
import logging

#Create and configure logger
logging.basicConfig(filename="people_counter.log",format='%(asctime)s %(message)s',filemode='a')

#Creating an object
log=logging.getLogger()

#Setting log level
log.setLevel(logging.DEBUG)

import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.58,
                        help="Probability threshold for detections filtering"
                        "(0.58 by default)")
    return parser


def draw_masks(result, frame, initial_w, initial_h, euclidean_distance, k):
        current_count = 0
        euclidean_distance = euclidean_distance
        for box in result[0][0]:
            # Draw bounding box for object when it's probability is more than the specified threshold
            conf = box[2]
            if conf > prob_threshold:
                xmin = int(box[3] * initial_w)
                ymin = int(box[4] * initial_h)
                xmax = int(box[5] * initial_w)
                ymax = int(box[6] * initial_h)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                current_count = current_count + 1

                # Center and mid point.
                c_x = frame.shape[1]/2
                c_y = frame.shape[0]/2
                mid_x = (xmax + xmin)/2
                mid_y = (ymax + ymin)/2

                # Calculating distance
                euclidean_distance =  math.sqrt(math.pow(mid_x - c_x, 2) +  math.pow(mid_y - c_y, 2) * 1.0)
                k = 0

        if current_count < 1:
            k += 1

        if euclidean_distance>0 and k < 10:
            current_count = 1
            k += 1
            if k > 100:
                k = 0

        return frame, current_count, euclidean_distance, k

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    model = args.model
    input_mode = args.input
    device  = args.device

    # Single image flag
    single_image_input_mode = False

    start_time = 0
    cur_request_id = 0
    last_count = 0
    total_count = 0
    duration = 0
    color = (255,0,0)
    temp_dist = 0
    tk = 0

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, device)
    net_input_shape = infer_network.get_input_shape()

    n, c, h, w = infer_network.load_model(model, device)[1]

    log.info("Input Dimensions of the loaded model {}{}{}{}".format(n,c,h,w))

    ### TODO: Handle the input stream ###
    # Live Camera feed
    if input_mode == 'CAMERA':
        input_stream = 0

    # Single Image
    elif input_mode.endswith('.jpg') or input_mode.endswith('.bmp') :
        single_image_input_mode = True
        input_stream = input_mode

    else:
        input_stream = input_mode
        assert os.path.isfile(input_mode), "Specified input file doesn't exist"

    try:
        cap=cv2.VideoCapture(input_stream)

    except FileNotFoundError:
        print("Cannot locate input stream file: "+ video_file)
    except Exception as e:
        print("Unknown error in input stream: ", e)

    global initial_w, initial_h, prob_threshold



    # Input frame width and height.
    width = cap.get(3)
    height = cap.get(4)
    prob_threshold = args.prob_threshold

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag,frame = cap.read()
        print("coming here")
        ### TODO: Pre-process the image as needed ###
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        log.info("Input frame size:- {}".format(frame.shape))
        pro_image = cv2.resize(frame,(w, h))
        log.info("resize frame shape:- {}".format(pro_image.shape))
        pro_image = pro_image.transpose((2, 0, 1))
        log.info("transposing frame:- {}".format(pro_image.shape))
        pro_image = pro_image.reshape((n,c,h,w))
        log.info("final processed image {}".format(pro_image.shape))
        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        log.info("starting the inference engine")
        infer_network.exec_net(pro_image)
        ### TODO: Wait for the result ###



        if infer_network.wait() == 0:
            log.info("Coming to infer network result section")
            det_time = time.time() - inf_start
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            ### TODO: Extract any desired stats from the results ###
            out_frame, current_count, dist, tk = draw_masks(result,frame, width, height, temp_dist, tk )
            # Printing Inference Time
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(out_frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

            # Calculate and send relevant information
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            if current_count < last_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))

            # Adding overlays to the frame
            txt2 = "Distance: %d" %dist + " Lost frame: %d" %tk
            cv2.putText(out_frame, txt2, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

            txt2 = "Current count: %d " %current_count
            cv2.putText(out_frame, txt2, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

            if current_count > 3:
                txt2 = "Alert! Maximum count reached"
                (text_width, text_height) = cv2.getTextSize(txt2, cv2.FONT_HERSHEY_COMPLEX, 0.5, thickness=1)[0]
                text_offset_x = 10
                text_offset_y = frame.shape[0] - 10
                # make the coords of the box with a small padding of two pixels
                box_coords = ((text_offset_x, text_offset_y + 2), (text_offset_x + text_width, text_offset_y - text_height - 2))
                cv2.rectangle(out_frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)

                cv2.putText(out_frame, txt2, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            client.publish("person", json.dumps({"count": current_count})) # People Count

            last_count = current_count
            temp_dist = dist
            # Display the resulting frame
            cv2.imshow('Output_Frame',out_frame)
            # Break if escape key is key_pressed
            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_input_mode:
            cv2.imwrite('output_image.jpg', out_frame)

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
