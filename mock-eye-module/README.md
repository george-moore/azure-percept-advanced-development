# Mock Eye Module

This directory contains all the code that we use to build and test a mock azureeyemodule application
using C++ and OpenVINO. The theory is that if the model works here, it should be easy to get working
on the Azure Percept DK.

## Building and Running

I recommend using Docker for this. To do so, you can simply use the scripts found in the `scripts` directory.

### Building

To just make sure the application compiles (without actually running it), you can simply run `./scripts/compile.sh`
or `./scripts/compile.ps1`. This should pull the appropriate Docker image if you don't already have it.

### Running

If you would like to compile and run the mock application, you can use `./scripts/compile_and_test.sh` or
`./scripts/compile_and_test.ps1` along with whatever arguments. See `./scripts/compile_and_test.sh --help` for help.

## Architecture and Extending

The architecture of the mocke eye module is quite simple. There is an enum in `modules/parser.hpp` with each
type of model that the mock application currently accepts. This is a subset of the models that the azureeyemodule
accepts, and this is on purpose to keep this application small and easy to use for porting to the Percept DK.

In `main.cpp`, there is a switch statement that looks at the value passed in on command line, matches it with one
of the available parsers and then hands control over to the parser.

If you want to use this for porting a new model to the Percept DK, you will want to take a look at the example parsers
and reference them when writing your own. See the [tutorials](../tutorials/README.md) in this repository for thorough
guidance.

When extending this application, please note a few folders:

* `kernels`: This folder contains all the OpenCV G-API kernels. You can, though you don't have to, add your kernels here.
* `modules/objectdetection`: This folder contains all the parsers for the object detectors in this mock application.
* `modules/openpose`: When extending this with something other than another object detector, you will likely want to
  create a new folder (and add it to the CMake file) and put all source code related to it inside that folder. In
  the case of Open Pose, which has very complicated post-processing logic, there are several files in this folder.
  This serves as an example for how you might want to structure your code, but try not to feel overwhelmed by its complexity,
  you don't need to understand anything about the logic in this folder to extend this application for your own model.