#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace cv;
using namespace std;

map <int, string> label_names;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    string line, path, classlabel;

    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }

    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);

        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
   	    label_names[atoi(classlabel.c_str())] = path.substr(path.find_first_of("/") 
		+ 1, path.find_last_of("/") - path.find_first_of("/") - 1);
        }
    }
}

int main(int argc, const char *argv[]) {
    if (argc != 4) exit(1);

    string fn_haar = string(argv[1]);   // Haar detector path
    string fn_csv = string(argv[2]);    // File path containing paths to the images
    int deviceId = atoi(argv[3]);       // Camera ID

    vector<Mat> images;
    vector<int> labels;

    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    // Keep size of the first image, so we can scale later to the same size
    int im_width = images[0].cols;
    int im_height = images[0].rows;

    // Face Recognition Model
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer(0, 123.0);
	
    // Train model with data
    model->train(images, labels);

    // Haar Cascade Classifier used to detect faces in camera input
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

    // Handle to camera device
    VideoCapture cap(deviceId);

    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << " cannot be opened." << endl;
        return -1;
    }
    
    // Camera frame
    Mat frame;

    while(true) {
        cap >> frame;
	
        // Clone the current frame
        Mat original = frame.clone();
        Mat gray;
	
        // Convert the current frame to grayscale
        cvtColor(original, gray, CV_BGR2GRAY);

        // Find the faces in the frame
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces, 2, 5);
 
        for(int i = 0; i < faces.size(); i++) {
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            Mat face_resized;

	    // Resize the face to model size
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

            int prediction_label = -1;    // Label of the predicted person
            double precision = 0.0;       // Certanity of the prediction

	    // Predict face
            model->predict(face_resized, prediction_label, precision);

            if (prediction_label == -1)  {
		rectangle(original, face_i, CV_RGB(255, 0, 0), 2);

                int pos_x = std::max(face_i.tl().x - 10, 0);
                int pos_y = std::max(face_i.tl().y - 10, 0);

                putText(original, "UNKNOWN", Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);
	        continue;
	    }

	    // Show the result
            rectangle(original, face_i, CV_RGB(0, 255, 0), 2);
    
	    string box_text = format("%s %f%", label_names[prediction_label].c_str(), precision); 
	
	    // Calculate position of the text to display
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            
            // Draw text
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }

        // Show the result
        imshow("face_recognizer", original);

        char key = (char) waitKey(27);   // Exit on ESC
        if(key == 27) break;
    }
    return 0;
}
