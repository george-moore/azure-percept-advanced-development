"""
Train the Pix2Pix model on the dreamified COCO dataset.
Taken (with modifications) from https://www.tensorflow.org/tutorials/generative/pix2pix under Apache 2.0.
"""
import argparse
import datetime
import os
import shutil
import time
import tensorflow as tf

LAMBDA = 100         # Pix2Pix uses a combination loss: discriminator + (lambda * L1 loss)
OUTPUT_CHANNELS = 3  # The number of output channels from the generator
IMG_WIDTH = 256      # Resize dataset images to ths width
IMG_HEIGHT = 256     # Resize dataset images to this height

def load(image_file):
    """
    Loads a single image and splits it into the original and the dreamified version.
    The combine_A_and_B.py script combines the raw images with the dreamified images
    by putting them together on the same x axis.
    """
    # Read in the image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # Get the width
    w = tf.shape(image)[1]

    # Find the halfway point on the x-axis and split into real and dreamified
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    # Convert the images to FP32
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def resize(input_image, real_image, height, width):
    """
    Resize the given images into height by width.
    """
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def normalize(input_image, real_image):
    """
    Normalizes the images into [-1.0, 1.0]. Assumes that the images were
    saved as uint8 data.
    """
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

def random_crop(input_image, real_image):
    """
    Crops the images randomly for image augmentation.
    """
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

@tf.function()
def random_jitter(input_image, real_image):
    """
    Apply some random jitter for image augmentation.
    """
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file):
    """
    Loads the image for training, by loading it, splitting it into
    dreamified and raw, then randomly jittering them and normalizing them.
    """
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    """
    Loads the image for testinging - does the same as loading for training,
    but does not do random jitter, only resizing.
    """
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def downsample(filters, size, apply_batchnorm=True):
    """
    This is a CNN downsample block used by our U-Net.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    """
    This is a CNN upsample block used by our U-Net.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def construct_generator():
    """
    This is the GAN generator portion. This is the model that we will use
    to create fancy images from seed images.
    """
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model and gathering the entries to the skip connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # Reverse the skip connections
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target, loss_object):
    """
    The loss for the generator is GAN_loss + (lambda * L1 loss),
    where the L1 loss is the L1 distance between the ground truth
    seed image and the image that the generator output.
    """
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def construct_discriminator():
    """
    Construct the discriminator model. This model's whole purpose
    is to help make the generator model better.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
    """
    """
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

@tf.function
def train_step(input_image, target, epoch, summary_writer, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer):
    """
                                   target_img ->|
    A single step of input_image -> Generator -> Discriminator -> backprop all the way back.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, loss_object)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, loss_object)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds, summary_writer, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer, checkpoint, checkpoint_prefix):
    """
    The training loop.
    """
    for epoch in range(epochs):
        start = time.time()

        # Train
        print("Epoch: ", epoch)
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image, target, epoch, summary_writer, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    checkpoint.save(file_prefix=checkpoint_prefix)

def train(dataset_dpath):
    """
    Trains a pix2pix model on the given dataset, returning the trained model.
    """
    BUFFER_SIZE = 999  # Take the whole dataset into memory if you can.
    BATCH_SIZE = 1
    EPOCHS = 150

    # If dataset_dpath does not have an ending slash, we need one
    if not dataset_dpath.endswith(os.path.sep):
        dataset_dpath += os.path.sep

    # Training dataset
    train_dataset = tf.data.Dataset.list_files(dataset_dpath + 'train/*.jpg')
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    # Testing dataset
    test_dataset = tf.data.Dataset.list_files(dataset_dpath + 'val/*.jpg')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Build the generator and the discriminator
    generator = construct_generator()
    discriminator = construct_discriminator()

    # Construct the loss for the discriminator
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Optimizers for each model
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Save checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    log_dir="logs/"
    summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    fit(train_dataset, EPOCHS, test_dataset, summary_writer, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer, checkpoint, checkpoint_prefix)

    return generator

if __name__ == "__main__":
    # !python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction AtoB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "-d", type=str, default="coco-dreamified", help="The root of the dataset.")
    parser.add_argumentd("--save", "-s", type=str, default="dream-pix2pix-tf", help="Directory where we will save our model.")
    args = parser.parse_args()

    # Check if the dataset exists
    if not os.path.isdir(args.dataset):
        print("Can't find the given dataset root. Given:", args.dataset)
        exit(1)

    # Train
    generator = train(args.dataset)
    dpath = args.save
    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    os.mkdir(dpath)
    generator.save(os.path.join(args.save, "dream-pix2pix-tf"))