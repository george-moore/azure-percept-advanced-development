"""
This script downloads the COCO dataset, then dreamifies N images from it
to create our own dataset made from deep-dreamifying images from COCO.
We use the COCO validation split to speed up download time, since
we don't need the whole COCO dataset.

Portions of this script were taken (with modifications) from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/deepdream.ipynb
under Apache 2.0.
"""
from tqdm import tqdm
import argparse
import os
import PIL.Image
import random
import wget
import zipfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nimgs", "-n", type=int, default=1000, help="Number of images to take from the COCO dataset.")
    parser.add_argument("--seed", type=int, default=2436, help="Random seed. We take images at random from COCO.")
    parser.add_argument("--coco", type=str, default=None, help="If given, should be a path to the COCO dataset, otherwise we will download it.")
    parser.add_argument("--destination", "-d", type=str, default="coco-dreamified", help="We will make this directory. This should either not exist or be empty if it does.")
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)

    # Assert that we have > 0 for our number of images
    if args.nimgs <= 0:
        print("Need --nimgs to be >= 0, but is", args.nimgs)
        exit(1)

    # Make sure the output directory is actually a directory if it exists
    if os.path.exists(args.destination) and not os.path.isdir(args.destination):
        print("There is a file called", args.destination, "and we need to make a directory called that.")
        exit(3)

    # Make sure the output directory is empty if it exists
    if os.path.isdir(args.destination) and os.listdir(args.destination):
        print("The directory we are putting our images into is not empty and it should be. Given", args.destination)
        exit(2)

    # Download COCO validation split
    cocopath = None
    if args.coco:
        cocopath = args.coco
    else:
        print("Downloading COCO validation split. This will take a few minutes on a reasonable internet connection.")
        url = "http://images.cocodataset.org/zips/val2017.zip"
        fname = wget.download(url)
        cocopath = "coco-validation-2017"
        with zipfile.ZipFile(fname, 'r') as zf:
            zf.extractall(cocopath)
        cocopath = os.path.join(cocopath, "val2017")

    # Read in all file paths
    original_img_fpaths = [os.path.join(os.path.abspath(cocopath), fname) for fname in os.listdir(cocopath)]

    # Randomly choose however many images from the file paths
    original_img_fpaths = [random.choice(original_img_fpaths) for _ in range(args.nimgs)]

    # Create an output location
    if not os.path.exists(args.destination):
        os.mkdir(args.destination)

    # For each file, dreamify it and save it into the location
    for inputfpath in tqdm(original_img_fpaths):
        dreamified_img = dreamify_image_at_path(inputfpath)

        name = os.path.basename(inputfpath)
        save_to_path = os.path.join(args.destination, name)
        dreamified_img.save(save_to_path)


# This script is taken (with modifications) from: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/deepdream.ipynb
# and retains the following license header:
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL.Image
from tensorflow.keras.preprocessing import image


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

                # Calculate the gradient of the loss with respect to the pixels of the input image.
                gradients = tape.gradient(loss, img)

                # Normalize the gradients.
                gradients /= tf.math.reduce_std(gradients) + 1e-8

                # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
                # You can update the image by directly adding the gradients (because they're the same shape!)
                img = img + gradients*step_size
                img = tf.clip_by_value(img, -1, 1)

        return loss, img

class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, img, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])

        ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[x:x+tile_size, y:y+tile_size]
                    loss = calc_loss(img_tile, self.model)

                    # Update the image gradients for this tile.
                    gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0,1])

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients


def download(url, max_dim=None):
    """
    Download an image and read it into a NumPy Array.
    """
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

def deprocess(img):
    """
    Normalize an image.
    """
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

def show(img):
    """
    Display an image.
    """
    plt.imshow(PIL.Image.fromarray(np.array(img)))
    plt.show()

def calc_loss(img, model):
    """
    Pass forward the image through the model to retrieve the activations.
    Converts the image into a batch of size 1.
    """
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return  tf.reduce_sum(losses)

def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))

        show(deprocess(img))
        print ("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    show(result)

    return result

def random_roll(img, maxroll):
    """
    Randomly shift the image to avoid tiled boundaries.
    """
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0,1])
    return shift, img_rolled

def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01, octaves=range(-2,3), octave_scale=1.3):
    get_tiled_gradients = TiledGradients(dream_model)
    base_shape = tf.shape(img)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        # Scale the image based on the octave
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img)
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)

            if step % 10 == 0:
                show(deprocess(img))
                print ("Octave {}, Step {}".format(octave, step))

    result = deprocess(img)
    return result

if __name__ == "__main__":
    url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

    def load_img(fpath, max_dim=None):
        img = PIL.Image.open(fpath)
        if max_dim:
            img.thumbnail((max_dim, max_dim))
        return np.array(img)

    # Downsizing the image makes it easier to work with.
    #original_img = download(url, max_dim=500)
    original_img = load_img("coco-validation-2017/val2017/000000129135.jpg", max_dim=500)

    show(original_img)
    show(download("https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"))

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    # Maximize the activations of these layers
    names = ['mixed3', 'mixed5']
    layers = [base_model.get_layer(name).output for name in names]

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    deepdream = DeepDream(dream_model)
    dream_img = run_deep_dream_simple(img=original_img, steps=100, step_size=0.01)

    shift, img_rolled = random_roll(np.array(original_img), 512)
    show(img_rolled)

    img = run_deep_dream_with_octaves(img=original_img, step_size=0.01)
    img = tf.image.resize(img, base_shape)
    img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
    show(img)
