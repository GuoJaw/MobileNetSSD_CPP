#include "ssd_detect.hpp"

#include <sstream>
#include <iostream>

const int isMobilenet = 1;

DEFINE_string(mean_file, "", "The mean file used to subtract from the input image.");

#if isMobilenet
DEFINE_string(mean_value, "127", "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
#else
DEFINE_string(mean_value, "104,117,123", "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
#endif

DEFINE_string(file_type, "image", "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "result/out.txt", "If provided, store the detection results in the out_file.");

DEFINE_double(confidence_threshold, 0.3, "Only store detections with score higher than the threshold.");  //置信度

int main(int argc, char** argv) {
        std::cout << "input ssd_main ..."  << std::endl;
        if (argc < 3) {
            std::cout << " Usage: classifier model_file weights_file list_file\n" << std::endl;
            return -1;
        }

        const string& model_file = argv[1];
        const string& weights_file = argv[2];
        const string& mean_file = FLAGS_mean_file;
        const string& mean_value = FLAGS_mean_value;

        const float confidence_threshold = FLAGS_confidence_threshold;

        // Initialize the network.
        Detector detector;
        detector.Set(model_file, weights_file, mean_file, mean_value, isMobilenet);

        cv::VideoCapture cap(argv[3]);
        if (!cap.isOpened())
        {
            std::cout << "Failed to open video: " << std::endl;
        }

        cv::Mat img;

        double fps = 0, t = 0.0;
        while (true)
        {
                bool success = cap.read(img);
                if (!success)
                {
                    break;
                }
                CHECK(!img.empty()) << "Error when read frame";

                //求FPS
                t = (double)cv::getTickCount();

                //MobileNet-SSD检测
                std::vector<vector<float> > detections = detector.Detect(img);
                detector.Postprocess(img, confidence_threshold, detections);

                //FPS
                t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
                fps = 1.0 / t;

                //将FPS画到图像上
                char str[20];
                sprintf(str, "%.2f", fps);
                std::string fpsString("FPS:");
                fpsString += str;
                putText(img, fpsString, cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
                std::cout << "FPS=" << fps << std::endl;

                //显示
                cv::imshow("ssd",img);
                if((char)cv::waitKey(1) == 'q')
                    break;
        }

        if (cap.isOpened()) {
            cap.release();
        }
        return 0;
}


