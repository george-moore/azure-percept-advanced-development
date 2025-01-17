// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// Third party includes
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

// Local includes
#include "helper.hpp"

namespace util {

static bool verbose_logging = false;

bool file_exists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}

void log_error(std::string str)
{
    time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << std::put_time(std::localtime(&t), "%Y-%m-%d %X") << " ERROR: " << str << std::endl;
}

void log_info(std::string str)
{
    time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << std::put_time(std::localtime(&t), "%Y-%m-%d %X") << " " << str << std::endl;
}

void log_debug(std::string str)
{
    if (verbose_logging)
    {
        time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::put_time(std::localtime(&t), "%Y-%m-%d %X") << " " << str << std::endl;
    }
}

std::string get_label(int index, const std::vector<std::string> &class_list)
{
    if ((index >= 0) && (static_cast<size_t>(index) < class_list.size()))
    {
        return class_list[index];
    }
    else
    {
        return std::to_string(index);
    }
}

void put_text(const cv::Mat &rgb, std::string text)
{
    cv::putText(rgb,
        text,
        cv::Point(300, 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(0, 0, 0),
        5);

    cv::putText(rgb,
        text,
        cv::Point(300, 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(255, 255, 255),
        2);
}

void set_logging(bool data)
{
    verbose_logging = data;
    log_info("verbose_logging: " + std::to_string(verbose_logging));
}

std::string to_hex_string(int i)
{
    std::stringstream ss;
    ss << std::hex << i;
    return ss.str();
}

std::string to_lower(std::string str)
{
    // convert string to back to lower case
    std::for_each(str.begin(), str.end(), [](char& c)
        {
            c = ::tolower(c);
        });

    return str;
}

std::string to_size_string(cv::Mat& mat)
{
    std::string str = std::string("");

    for (int i = 0; i < mat.dims; ++i)
    {
        // Append x234 (for example), but only if it is not the first dim
        str.append(i ? "x" : "").append(std::to_string(mat.size[i]));
    }

    return str;
}

std::string to_string_with_precision(float f, int precision)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << f;
    return ss.str();
}

int run_command(std::string command)
{
    log_info(command);
    return system(command.c_str());
}

bool search_keyword_in_file(std::string keyword, std::string filename)
{
    std::ifstream file(filename);

    if (file.is_open())
    {
        std::string line;

        while (getline(file, line))
        {
            if (std::string::npos != line.find(keyword))
            {
                return true;
            }
        }

        file.close();
    }

    return false;
}

void version()
{
    system("stat -c 'Inference App Version: %y' /app/inference | cut -c -33");
}

std::vector<std::string> splice_comma_separated_list(const std::string &list_string)
{
    std::vector<std::string> result;
    std::stringstream ss(list_string);

    while (ss.good())
    {
        std::string substr;
        getline(ss, substr, ',');
        result.push_back(substr);
    }

    return result;
}

} // namespace util
