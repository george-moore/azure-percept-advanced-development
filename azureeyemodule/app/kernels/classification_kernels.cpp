// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/streaming/desync.hpp>

#include "classification_kernels.hpp"


namespace cv {
namespace gapi {
namespace streaming {

/** C++ wrapper for classification op */
GClassificationsWithConf parseClass(const GMat& in, const GOpaque<Size>& in_sz, float confidence_threshold)
{
    return GParseClass::on(in, in_sz, confidence_threshold);
}

} // namespace streaming
} // namespace gapi
} // namespace cv
