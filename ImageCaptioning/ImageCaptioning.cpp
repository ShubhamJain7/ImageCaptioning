#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const char* vocab_file_path = "C:/Users/dell/source/repos/ImageCaptioning/data/vocab.txt";
const string image_path = "C:/Users/dell/source/repos/ImageCaptioning/test.jpg";

Mat preprocess_image(Mat image) {
    // convert image values from int to float
    image.convertTo(image, CV_32FC3);
    // Change image format from BGR to RGB
    cvtColor(image, image, COLOR_BGR2RGB);

    Mat image_resized;
    // resize image to (256x256) to fit model input dimensions
    resize(image, image_resized, Size(256, 256));

    // normalize image (values between 0-1)
    Mat image_float;
    image_resized.convertTo(image_float, CV_32FC3, 1.0f / 255.0f, 0);

    // split image channels
    vector<cv::Mat> channels(3);
    split(image_float, channels);

    // define mean and std-dev for each channel
    vector<double> mean = { 0.485, 0.456, 0.406 };
    vector<double> stddev = { 0.229, 0.224, 0.225 };
    size_t i = 0;
    // normalize each channel with corresponding mean and std-dev values
    for (auto& c : channels) {
        c = (c - mean[i]) / stddev[i];
        ++i;
    }

    // concatenate channels to change format from HWC to CHW
    Mat image_normalized;
    vconcat(channels, image_normalized);

    return image_normalized;
}

int main()
{
    // declare vector to store all words
    vector<string> vocab;
    // open vocab file
    ifstream file(vocab_file_path);
    string line;
    // read vocab file line by line and append to vocab vector
    while (std::getline(file, line)) {
        vocab.push_back(line);
    }

    // load image to process
    Mat im;
    im = imread(image_path, IMREAD_COLOR);
    // if image is an empty matrix
    if (!im.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // get processed image
    Mat image = preprocess_image(im);
}
