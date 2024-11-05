import os
import numpy as np
import argparse
import logging
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import Model
from keras.preprocessing import image
from waggle.plugin import Plugin
from waggle.data.vision import Camera

TOPIC = "env.binary.flood"
LABELS = ['Flooding', 'No Flooding']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="bottom",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-model', dest='model',
        action='store', default='flood_detection_model.keras',
        help='Path to model')
    parser.add_argument(
        '-continuous', dest='continuous',
        action='store_true', default=False,
        help='Continuous run flag')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='enable debug logs')

    return parser.parse_args()

def preprocess_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def run(model, sample, do_sampling, plugin):
    image = sample.data
    timestamp = sample.timestamp

    img = preprocess_image(image)
    # if args.device == 'cpu': TODO: enable GPU access
    #     img = img.to(device='cpu')
    # elif args.device == 'cuda':
    #     img = img.to(device='cuda')
    
    pred = model.predict(img)
    result = np.argmax(predictions)

    plugin.publish(TOPIC, result, timestamp=timestamp)
    msg = f"run(): {LABELS[result]} at time: {timestamp}"
    logging.debug(msg)

    if do_sampling:
        sample.data = image
        sample.save('sample.jpg')
        plugin.upload_file('sample.jpg')
        logging.debug('run(): saved sample')

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    # if torch.cuda.is_available(): TODO: enable GPU access
    #     args.device = 'cuda'
    #     model = torch.load(args.model, map_location='cuda')
    # else:
    #     args.device = 'cpu'
    #     model = torch.load(args.model, map_location='cpu')
    # model.eval()
    logging.info('__main__: Model is loading...')
    model = keras.models.load_model(args.model)
    logging.info('__main__: Model was loaded')

    sampling_countdown = -1
    if args.sampling_interval >= 0:
        sampling_countdown = args.sampling_interval

    while True:
        with Plugin() as plugin, Camera(args.stream) as camera:
            sample = camera.snapshot()

            do_sampling = False
            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                do_sampling = True
                sampling_countdown = args.sampling_interval
            logging.info('__main__: Predictions started')
            run(model, sample, do_sampling, plugin)
            if not args.continuous:
                exit(0)   ## oneshot