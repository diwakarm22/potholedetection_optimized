#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h> // PyTorch C++ API

using namespace std;
using namespace cv;
using namespace torch;

int main() {
    // Load YOLOv8 model
    const string model_path = "runs/detect/train/weights/best.pt";
    torch::jit::script::Module yolo_model = torch::jit::load(model_path);

    // Open the video file
    VideoCapture cap("pothole_video.mp4");
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open video file." << endl;
        return -1;
    }

    while (true) {
        // Read a frame from the video
        Mat frame;
        cap.read(frame);

        // Perform object detection on the frame
        // Convert the frame to tensor format (assuming the frame is in BGR format)
        Mat frame_rgb;
        cvtColor(frame, frame_rgb, COLOR_BGR2RGB);
        tensor::Tensor tensor_image = torch::from_blob(frame_rgb.data, {1, frame_rgb.rows, frame_rgb.cols, 3}, torch::kByte);
        tensor_image = tensor_image.permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;
        tensor_image = tensor_image.to(torch::kCPU);

        // Execute the YOLO model
        tensor::Tensor detections = yolo_model.forward({tensor_image}).toTensor();

        // Post-process the detections (you need to implement this part based on your YOLO model output)
        // For simplicity, I'm just displaying the original frame here
        imshow("YOLO V8 Detection", frame);

        // Check for user input to exit (press any key to exit)
        if (waitKey(1) != -1)
            break;
    }

    // Release video capture object and close all windows
    cap.release();
    destroyAllWindows();

    return 0;
}
