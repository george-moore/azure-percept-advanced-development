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