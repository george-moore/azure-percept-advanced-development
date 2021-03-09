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

To stop it, hit CTRL-C and then run `docker ps` and `docker stop <whichever-container>`.

If you are curious, you could try the same thing using OpenVINO Model Zoo's `faster_rcnn_resnet50_coco` (parser argument "faster-rcnn"),
`yolo-v2-tiny-tf` (parser argument "yolo"), or OpenPose (parser argument "openpose"). The first two can be downloaded from the workbench, the OpenPose model
can be [downloaded from here](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/human-pose-estimation-0001/FP32/).
If you use any of these, you will need to give a --parser argument.

## Semantic Segmentation

Now that we've got a sandbox to test our model in, let's work on porting the semantic segmentation model over to it.

The first thing to do is to use the OpenVINO Workbench again. This time, instead of downloading SSD, search for
"semantic", and that should be good enough to find "semantic-segmentation-adas-0001". Import it, download it, and extract it
into its two files. Put the files somewhere where you won't lose them.

Now let's go through what you will need to do to add support for this model to the mock-eye-app. Remember the point of adding
support for this model to the mock-eye-app is that doing so will put us about halfway towards our real goal of porting
this model to the Percept application instead. This sandbox application will allow us to run GDB and to feed a movie file
as an input to the application, while also being a much smaller application that is easier to reason about.

Here are the steps that are needed to add support to the mock-eye-app:

1. Add a new "parser" variant to the enum in mock-eye-app/modules/parser.[c/h]pp,
   and don't forget to update the `look_up_parser()` function there so the command line can accept your parser as an argument.
1. Add a folder under `modules` called `segmentation`, and then update the CMakeLists.txt file to include the new folder.
1. Put all of our runtime logic in the `modules/segmentation` folder, which will include compiling a G-API graph,
   passing the graph our custom kernels (which we will make in this tutorial), and then running the graph, collecting
   the graph's outputs, and interpreting them.
1. Implement whatever custom kernels we need for our G-API graph.

Once we have completed these steps, porting the resulting logic to the Percept DK should be pretty simple, meanwhile, we'll complete all
of these steps on our host PC, which should make development quite a bit more comfortable.

Let's go through these steps one at a time.

### Parser Enum

In order to integrate our new model (and its post-processing logic - i.e., "parser") into the mock app, we need to tell the command line
arguments and the main function that we have a new AI model that we can accept.

Let's start by updating the enum and look-up function in `mock-eye-module/modules/parser.hpp` and `mock-eye-module/modules/parser.cpp`:

First, here's the .hpp file:

```C++
// The contents of this enum may have changed by the time you read this,
// because maybe I forgot to update this documentation. But either way,
// find this enum and update it to include "UNET_SEM_SEG" or whatever you want to call it.
enum class Parser {
    FASTER_RCNN,
    OPENPOSE,
    SSD100,
    SSD200,
    // Here's the new item
    UNET_SEM_SEG,
    YOLO
};
```

Next, here's the .cpp file:

```C++
Parser look_up_parser(const std::string &parser_str)
{
    if (parser_str == "openpose")
    {
        return Parser::OPENPOSE;
    }
    else if (parser_str == "ssd100")
    {
        return Parser::SSD100;
    }
    else if (parser_str == "ssd200")
    {
        return Parser::SSD200;
    }
    else if (parser_str == "yolo")
    {
        return Parser::YOLO;
    }
    else if (parser_str == "faster-rcnn")
    {
        return Parser::FASTER_RCNN;
    }
    else if (parser_str == "unet-seg") ///////// This is the new one
    {                                  /////////
        return Parser::UNET_SEM_SEG;   /////////
    }                                  /////////
    else
    {
        std::cerr << "Given " << parser_str << " for --parser, but we do not support it." << std::endl;
        exit(-1);
    }
}
```

Now update main.cpp:

```C++
// .... other code

/** Arguments for this program (short-arg long-arg | default-value | help message) */
static const std::string keys =
"{ h help    |        | Print this message }"
"{ d device  | CPU    | Device to run inference on. Options: CPU, GPU, NCS2 }"
"{ p parser  | ssd100 | Parser kind required for input model. Possible values: ssd100, ssd200, yolo, openpose, faster-rcnn, unet-seg }" // Update the help message
"{ w weights |        | Weights file }"
"{ x xml     |        | Network XML file }"
"{ labels    |        | Path to the labels file }"
"{ show      | false  | Show output BGR image. Requires graphical environment }"
"{ video_in  |        | If given, we use this file as input instead of the camera }";

// .... clip some more code

// Here we are in main():
    std::vector<std::string> classes;
    switch (parser)
    {
        case parser::Parser::OPENPOSE:
            pose::compile_and_run(video_in, xml, weights, dev, show);
            break;
        case parser::Parser::SSD100:  // Fall-through
        case parser::Parser::SSD200:  // Fall-through
        case parser::Parser::YOLO:
            classes = load_label(labelfile);
            detection::compile_and_run(video_in, parser, xml, weights, dev, show, classes);
            break;
        case parser::Parser::FASTER_RCNN:
            classes = load_label(labelfile);
            detection::rcnn::compile_and_run(video_in, xml, weights, dev, show, classes);
            break;
        case parser::Parser::UNET_SEM_SEG:                                        // NEW CODE
            classes = load_label(labelfile);                                      // NEW CODE
            semseg::compile_and_run(video_in, xml, weights, dev, show, classes);  // NEW CODE
            break;                                                                // NEW CODE
        default:
            std::cerr << "Programmer error: Please implement the appropriate logic for this Parser." << std::endl;
            exit(__LINE__);
    }
```

So now we've updated all the logic we need to route the application's flow to the right place if the user executes
this application with the `--parser unet-seg` argument.

Of course, this won't compile yet, since we don't have a `semseg` at all, let alone a `compile_and_run` function in it.
So let's code up that function now.

### Modules/Segmentation

Create a folder where we will put all of our semantic segmentation code: `mkdir modules/segmentation`.

Now let's create the header file, which will be entirely boilerplate:

```C++
// Put this in a file called mock-eye-module/modules/segmentation/unet_semseg.hpp

/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#pragma once

// Standard library includes
#include <string>
#include <vector>

// Our includes
#include "../device.hpp"
#include "../parser.hpp"

namespace semseg {

/**
 * Compiles the GAPI graph for a semantic segmentation model (U-Net, specifically) and runs the application. This method never returns.
 *
 * @param video_fpath: If given, we run the model on the given movie. If empty, we use the webcam.
 * @param modelfpath: The path to the model's .xml file.
 * @param weightsfpath: The path to the model's .bin file.
 * @param device: What device we should run on.
 * @param show: If true, we display the results.
 * @param labels: The labels this model was built to detect.
 */
void compile_and_run(const std::string &video_fpath, const std::string &modelfpath, const std::string &weightsfpath, const device::Device &device, bool show, const std::vector<std::string> &labels);

} // namespace semseg
```

How did I know that that's the code I should put in the header? By looking at `modules/objectdetection/faster_rcnn.hpp`.

Each of the arguments to the single function that we need to implement is explained in the header file, but
to be more verbose:

* `video_fpath`: Must be a valid path to a video file. Note that on Windows, the webcam won't work - only video files are supported.
* `modelfpath`: The path to the model's .xml file. Remember that each model is in the OpenVINO IR format, and therefore is composed
  of a topology (.xml) file and a weights (.bin) file.
* `weightsfpath`: The path to the model's .bin file.
* `device`: We haven't talked about devices. If you are curious, you can check in the device module, but the gist of it is that since the Inference Engine
  OpenCV back end we use in this application supports GPUs, Myriad X VPUs, and CPUs, I figured we could just support all of them. Unfortunately for Windows
  users, only CPU is supported.
* `show`: We don't need to show the GUI, but it is cool (and helpful for debugging). You could certainly get a way with just using
  `std::cout` messages.
* `labels`: Our U-Net model is trained to do semantic segmentation on particular items. If we don't give this over to the function,
  we'll make sure that the function just displays numbers instead of labels, so it is technically optional. Nonetheless, we'll pass something in either way,
  and if the function can't find the given file (perhaps because it is just an empty string, and not a file path at all), then we'll ignore this arg
  and output numbers instead of letters.

Let's add the .cpp file now.

```C++

```

## Label file

Since our semantic segmentation network is going to be classifying objects it sees and coloring
them, it should also write what they are as well. In order to do that, it will need a label file.
I didn't create this network, so I don't know what the labels are which correspond to the outputs,
but OpenVINO does: put the following into a labels.txt file.

```
road
sidewalk
building
wall
fence
pole
traffic light
traffic sign
vegetation
terrain
sky
person
rider
car
truck
bus
train
motorcycle
bicycle
ego-vehicle
```