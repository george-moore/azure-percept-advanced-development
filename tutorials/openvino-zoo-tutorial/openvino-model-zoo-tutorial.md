# OpenVINO Model Zoo Tutorial

**Please note!** The experiences in this repository should be considered to be in **preview/beta**.
Significant portions of these experiences are subject to change without warning. **No part of this code should be considered stable**.
Although this device is in public preview, if you are using these advanced experiences, we would like your feedback. If you have not
[registered for *private* preview](https://go.microsoft.com/fwlink/?linkid=2156349), please consider doing so before using this functionality
so that we can better track who is using it and how, and so that you can have a say in the changes we make to these experiences.

This tutorial shows you how to start from an OpenVINO model zoo neural network that has already been trained to do something
fairly general, and then port that model to the Azure Percept DK.

Once you have completed this tutorial, you will know how to port a model to the device. After that, you should consider
popping over to the [PyTorch model porting tutorial](../pytorch-from-scratch-tutorial/pytorch-from-scratch-tutorial.md), which will tell you how to train
up a brand new model and then port it to the device.

## Select a Model

If you have a need for a visual AI pipeline that can be solved by a model found in a zoo, such as general object detection,
person detection, etc., you can take a look at [the OpenVINO model zoo](https://docs.openvinotoolkit.org/2021.1/omz_models_intel_index.html).

For the purposes of this tutorial, we will be working with a simple [semantic segmentation model](https://docs.openvinotoolkit.org/2021.1/omz_models_intel_semantic_segmentation_adas_0001_description_semantic_segmentation_adas_0001.html).

You can see from that page some basic information about this model:

* It was trained on 20 classes (all of which seem to be outside/on the road).
* Its input is an image of size 2048x1024 (scrolling down the page to the bottom, you can see it really requires
  an input of shape [batch size, n-channels, 1024, 2048]).
* It has 6.686 million parameters.

Let's talk about a few things first.

### Input Format

The Azure Percept DK takes in images, one at a time, from the camera sensor. The camera sensor is wide-angled and high resolution,
but its output gets preprocessed a bit before we feed it into neural networks, according to the following table.

![Percept DK image preprocessing](imgs/preprocessing.png "Percept DK image preprocessing")

The table shows that we currently support three resolution modes: native, 1080p, and 720p. Either way, the image is ultimately resized
to whatever image size is expected by the network. So we don't really need to worry about the size of the input image.

### Network Size

We use an Intel Myriad X VPU for neural network acceleration on the Azure Percept DK. If your neural network is too large,
it won't fit in the VPU memory, and your network will run extremely slowly.

Specifically, the device has an upper limit of 165 MB for a neural network .blob file.

Note that this device does not allow for int8 quantization schemes, though we can use FP16. Let's assume
we go with FP32 just to see though, since this network is 6.686x10^6 parameters,
multiply this by four bytes per parameter, and we get 26,744,000 bytes, or about 26 MB. This network is easily small
enough to fit in our device, even if we use FP32. FP16 will halve that size.

## Example Code

While we've tried to make the experience of bringing your own AI model pipeline to the Azure Percept DK as easy as possible,
it is the nature of these models that some code will have to be written. In particular, the post-processing of the neural network
is something that you will have to take care of, because we can't know ahead of time what you will want to do with a network's output.
This will be a lot easier if you have some example code to go off of.

So, let's find some sample code for this model. Oh look, [here it is](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/segmentation_demo_async/main.cpp).
Note the license is Apache 2.0.

Let's try creating a super simple OpenCV application that tests this model out. First, let's just run the sample as is, then we'll strip it
down to just the part(s) that we care about.

To try running this sample unfortunately requires installing OpenVINO, which is outside the scope of this tutorial, as it is not
really necessary to port a model to the device. If you end up using a lot of models from the Intel Open Model Zoo, you should
definitely do this, but otherwise just take it from me that this sample does in fact work.

Let's instead integrate this example into our mock-eye-module application.

## Mock Eye Module

At the root of this repository is a [mock-eye-module](../../mock-eye-module/README.md) folder, which is useful for porting models from
PC to the Azure Percept DK. You could instead port the model directly into the azureeyemodule application and deploy it onto the device,
but, being an embedded device, the Azure Percept DK is not a great tool for going through debugging cycles.

So let's port this main.cpp file into the Mock Eye Module and make sure it works there before we port it all the way over to the device.

If you look at the mock-eye-module directory, you can see it contains a C++ application that can be compiled and run using
Docker. Let's build it and run it before making any changes to it, just to make sure it works.

### Prerequisites

Whether you are on Linux/OSX or Windows, you should be able to run the OpenVINO workbench to procure an SSD model, which
we will use for testing the mock eye module.

```bash
cd ../../scripts
./run_workbench.sh # if you are on windows, you can use ./run_workbench.ps1 instead
```

Follow the link it displays to the console if it does not automatically open a new tab in a browser.

1. Once in the browser, accept the cookies and push "create" in the top left to create a new configuration.
1. Now click "import" in the middle of the page.
1. Under the "Open Model Zoo" tab, search for ssd_mobilenet_v2_coco and select it and then "import and download".
1. FP16 or FP32 are both acceptable (when doing this for the device, you will generally want FP16). Push "Convert".
1. Once it completes, you may click the download arrow on the far right.
1. Extract the tar.gz archive to get the two files: `ssd_mobilenet_v2_coco.bin` and `ssd_mobilenet_v2_coco.xml`. These
   two files comprise what is called the OpenVINO IR (or intermediate representation) format of this model.
   The XML file contains the topography of the model, and the .bin file contains the weight values.
1. Put these two files wherever is convenient.

Now go and find an mp4 file from somewhere. I haven't uploaded one to GitHub, but you can find all kinds of movie files
out there under whatever licenses.

### Unix

If you are on Linux (I don't have Mac, so I can't test it, but it probably works too),

```bash
# Make sure you are in the mock-eye-module directory
cd ../../mock-eye-module
./scripts/compile_and_test.sh --video=<path to the video file> --weights=<path to the .bin> --xml=<path to the .xml>
```

### Windows

Unfortunately, there is some additional set up for Windows. See [the appropriate README](../../mock-eye-module/README.md)
for instructions. You will need to install VcXsrv and then launch it. Because you will be getting your XWindow GUIs
over an XWindow server, you will need to give the Docker container your IP address.

```powershell
cd ../../mock-eye-module
# You don't need the .bin file here because it figures it out from the .xml file, as long as they have
# the same name and are in the same folder.
# Why didn't I do this for the bash script too? I don't know.
./scripts/compile_and_test.ps1 -ipaddr <your IP address> -xml <path-to-the-xml> -video <path-to-the-video>
```

Whatever OS you are running, the result should be the same. The script should create a tmp folder, copy all the
source code into it, copy the .mp4 file into it, copy the model files into it, then launch a Docker container
that compiles and runs the application. You should see a GUI display the video, with bounding boxes overlayed on
it whenever the SSD network detects something above a certain confidence threshold.

We didn't bother feeding a labels file into this, so the labels are just numbers. It doesn't matter though,
since the point is just to make sure it works.

If you are curious, you could try the same thing using OpenVINO Model Zoo's `faster_rcnn_resnet50_coco` (parser argument "faster-rcnn"),
`yolo-v2-tiny-tf` (parser argument "yolo"), or OpenPose (parser argument "openpose"). The first two can be downloaded from the workbench, the OpenPose model
can be [downloaded from here](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/human-pose-estimation-0001/FP32/).
If you use any of these, you will need to give a --parser argument.

## Semantic Segmentation

Now that we've got a sandbox to test our model in, let's work on porting the semantic segmentation model over to it.

The first thing to do is to use the OpenVINO Workbench again. This time, instead of downloading SSD, search for
"semantic", and that should be good enough to find "semantic-segmentation-adas-0001". Import it, download it, and extract it
into its two files.