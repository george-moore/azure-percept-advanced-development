# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

<#
  .SYNOPSIS
  Compiles and runs the mock Azure Eye Percept DK application.

  .DESCRIPTION
  Compiles and runs the mock application, accepting a bunch of command line
  arguments. This script will create a temporary folder, copy all the source
  code into it, then run a Docker container that maps that folder into the
  Docker container's filesystem. Inside the container, we compile the source
  code and run the resulting application.

  .PARAMETER Debug
  If given, we compile the application in Debug mode and drop you into a GDB session.

  .PARAMETER Device
  Must be one of "NCS2", "GPU", or "CPU". Defaults to "CPU".
  NCS2 is an Intel Neural Compute Stick 2, which should be plugged into the USB slot on the PC.
  GPU is a supported GPU.

  .PARAMETER Labels
  The label file for the neural network. Not all models will require this.

  .PARAMETER Parser
  The string representation of the parser enum variant you want to use. See the application's
  help for accepted variants. Defaults to SSD.

  .PARAMETER Video
  If given, should be a path to a .mp4 file, which will serve as the input video source.
  If not given, we attempt to use your PC's webcam.

  .PARAMETER Xml
  The path to the .xml file of the neural network, which must be in OpenVINO IR format,
  and therefore must be composed of an .xml file and a .bin file. The .xml file and .bin
  file should have the same name other than the extensions, and should be in the same folder.

  .INPUTS
  None. You cannot pipe objects into this script.

  .OUTPUTS
  None.
#>

param (
    [Alias('g')][switch]$debug = $False,
    [Alias('d')][ValidateSet("NCS2", "GPU", "CPU")][string]$device = "CPU",
    [Alias('l')][ValidateScript({Test-Path $_ -PathType "Leaf"})][string]$labels,
    [Alias('p')][string]$parser = "ssd100",
    [Alias('v')][ValidateScript({Test-Path $_ -PathType "Leaf"})][string]$video,
    [Alias('x')][ValidateScript({Test-Path $_ -PathType "Leaf"})][string]$xml
)

# Exit on errors
$ErrorActionPreference = "Stop"

# Make sure we are in the right directory
If (-NOT (Test-Path "main.cpp" -PathType "Leaf")) {
    Write-Host "You must run this script from the mock-eye-module directory."
    Exit 1
}

# Create a $bin variable analagous to $xml
$bin = [io.path]::GetFileNameWithoutExtension($xml) + ".bin"

# Map the variables from where they are on the host system to the Docker container
$labelfile = "/home/openvino/tmp/labels.txt"
$videofile = ""
$weightfile = "/home/openvino/tmp/model.bin"
$xmlfile = "/home/openvino/tmp/model.xml"
$videofile = "/home/openvino/tmp/movie.mp4"

# Now create the tmp directory and copy everything into it
New-Item -Type Dirctory -Name tmp
Copy-Item -Recurse kernels tmp/
Copy-Item -Recurse modules tmp/
Copy-Item main.cpp tmp/
Copy-Item CMakeLists.txt tmp/

# For each of these files, if they exist, copy them into the temp directory.
if (Test-Path $labels) {
    $path = "tmp/labels.txt"
    Copy-Item $labels $path
}

if (Test-Path $video) {
    $path = "tmp/movie.mp4"
    Copy-Item $video $path
}

if (Test-Path $xml) {
    $path = "tmp/model.xml"
    Copy-Item $xml $path
}

if (Test-Path $bin) {
    $path = "tmp/model.bin"
    Copy-Item $bin $path
}

# Put together the command, based on whether we want webcam or not.
$appcmd = "./mock_eye_app --device " + $device + " --parser=" + $parser + " --weights=" + $weightfile + " --xml=" + $xmlfile + " --show"
if (Test-Path $video -PathType "Leaf") {
    $appcmd += " --video_in=" + $videofile
}

# Should we debug?
$debug_docker_cmd = ""
$buildtype = ""
if ($debug) {
    $appcmd = "gdb --args " + $appcmd
    $debug_docker_cmd = "-it"
    $buildtype = "Debug"
} else {
    $debug_docker_cmd = "-t"
    $buildtype = "Release"
}

$dockercmd =  "openvino/ubuntu18_runtime:2021.1 bash -c "
$dockercmd += "source /opt/intel/openvino/bin/setupvars.sh && "
$dockercmd += "mkdir -p build && "
$dockercmd += "cd build && "
$dockercmd += "cmake -DCMAKE_BUILD_TYPE=" + ${buildtype} + " .. && "
$dockercmd += "make && "
$dockercmd += $appcmd
Write-Host $dockercmd

$tmppath = $(Resolve-Path tmp).Path
docker run --rm `
            -e DISPLAY=${DISPLAY}  `
            -v /dev/bus/usb:/dev/bus/usb `
            -v ${tmppath}:/home/openvino/tmp `
            -w /home/openvino/tmp `
            --device=/dev/video0:/dev/video0 `
            --device=/dev/dri:/dev/dri `
            --device-cgroup-rule='c 189:* rmw' `
            --network=host `
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" `
            ${$debug_docker_cmd} `
            ${dockercmd}
