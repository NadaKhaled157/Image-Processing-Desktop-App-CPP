#include "image_processing.h"
#include "ui_image_processing.h"

using namespace std;

Image_Processing::Image_Processing(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    this->setWindowTitle("PixFusion");
    this->setWindowIcon(QIcon(":/Image_Processing/Resources/app_icon.png"));



    qDebug() << "Application Running.";
    /*cout << "Application Running.";*/

    ui->slider_param_1->setRange(0, 100);
    ui->slider_param_2->setRange(0, 100);
    ui->slider_param_1->setValue(noiseParam_1);
    ui->slider_param_2->setValue(noiseParam_2);

    ui->Widget_Org_Image->installEventFilter(this); // widget responsible for image upload

    //  Noise Combo Box
    connect(ui->Combox_Noise, QOverload<int>::of(&QComboBox::currentIndexChanged),
        [this](int index) {
            bool comingFromFilter = false;
            onNoiseComboBoxSelectionChanged(index, comingFromFilter);
        });


    // Noise Sliders
    connect(ui->slider_param_1, &QSlider::valueChanged, this, &Image_Processing::onSliderValueChanged);
    connect(ui->slider_param_2, &QSlider::valueChanged, this, &Image_Processing::onSliderValueChanged);

    // Filter Combo Box
    connect(ui->Combox_Filter, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, &Image_Processing::onFilterComboBoxSelectionChanged);

    // Edge Detector Combo Box
    connect(ui->Combox_Edges, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, &Image_Processing::onEdgeComboBoxSelectionChanged);

    // Histogram & Distribution Radio Button
    connect(ui->RadioButton_Histogram, &QRadioButton::toggled, this, &Image_Processing::drawHistogramAndDistribution);
    // Gray Image Histogram
    connect(ui->checkBox_grayImg, &QCheckBox::checkStateChanged, this, &Image_Processing::drawHistogramAndDistribution);

    // Normalize and Equalize
    connect(ui->RadioButton_Normalizer, &QRadioButton::toggled, this, &Image_Processing::normalizeAndEqualize);

    // Local and Global Thresholding
    connect(ui->RadioButton_Threshold, &QRadioButton::toggled, this, &Image_Processing::onThresholdingSelected);
    
}

Image_Processing::~Image_Processing() {
    delete ui;
}

bool Image_Processing::eventFilter(QObject* obj, QEvent* event) {
    if (obj == ui->Widget_Org_Image && event->type() == QEvent::MouseButtonDblClick) {
        // Double-click detected -> Load image
        QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Images (*.png *.jpg *.bmp)");
        if (!fileName.isEmpty()) {
            originalImageColored = cv::imread(fileName.toStdString()); // Read image normally
            originalImage = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE); // Read image grayscale
            if (!originalImage.empty()) {

                showImage(originalImage, ui->Widget_Org_Image);  // Display image
                // Reset Options
                ui->Combox_Noise->setCurrentIndex(0);
                ui->Combox_Filter->setCurrentIndex(0);
                ui->Combox_Edges->setCurrentIndex(0);
                // Clear outputs upon new upload
                ui->Widget_Output_1->clear();
                ui->Widget_Output_2->clear();
                qDebug() << "Image uploaded successfully.";
            }
            else {
                qDebug() << "Failed to load image.";
            }
        }
        return true;
    }
    return QMainWindow::eventFilter(obj, event);
}

void Image_Processing::showImage(const cv::Mat& img, QLabel* imageLabel) {
    if (img.empty()) {
        qDebug() << "Empty image!";
        return;
    }

    if (!imageLabel) {
        qDebug() << "Error: QLabel is nullptr.";
        return;
    }

    // Convert OpenCV Mat to QImage
    QPixmap pixmap;

    if (img.channels() == 3) {
        QImage qimg(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888);
        pixmap = QPixmap::fromImage(qimg.rgbSwapped()); // Convert BGR to RGB
    }
    else {

        QImage qimg(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8);
        pixmap = QPixmap::fromImage(qimg).scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    

    // Get QLabel size without forcing expansion
    QSize labelSize = imageLabel->size();
    QPixmap scaledPixmap = pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    

    // Set image without changing QLabel policy
    imageLabel->setPixmap(scaledPixmap);
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setScaledContents(false); // Prevent QLabel from expanding
}

// ADDING NOISE //

void Image_Processing::onNoiseComboBoxSelectionChanged(int index, bool comingFromFilter) {

    if (originalImage.empty()) {
        qDebug() << "No image uploaded!";
        return;
    }

    QString selectedNoise = ui->Combox_Noise->itemText(index);
    qDebug() << "Selected Noise:" << selectedNoise;
    ui->Label_output_1->setText("Noisy Image");
    ui->Label_output_2->setText("Filtered Image");

    if (selectedNoise == "Gaussian") {
        ui->label_param_1->setText("         Mean           ");
        ui->label_param_2->setText("Standard Deviation");
        noisyImage = addGaussianNoise(originalImage, noiseParam_1, noiseParam_2);
    }
    else if (selectedNoise == "Uniform") {

        ui->label_param_1->setText("Lower Bound");
        ui->label_param_2->setText("Upper Bound");
        noisyImage = addUniformNoise(originalImage, noiseParam_1, noiseParam_2);
        /*ui->slider_param_1->setRange(0, 100);
        ui->slider_param_2->setRange(0, 100);
        ui->slider_param_1->setValue(lowerBound);
        ui->slider_param_2->setValue(upperBound);
        noisyImage = addSaltPepperNoise(originalImage, lowerBound, upperBound);*/

    }
    else if (selectedNoise == "Salt and Pepper") { // Salt and Pepper
        cout << "ADDING SALT AND PEPPER NOISE";
        ui->label_param_1->setText("   Salt   ");
        ui->label_param_2->setText("Pepper");
        noisyImage = addSaltPepperNoise(originalImage, noiseParam_1, noiseParam_2);
        /*ui->slider_param_1->setRange(0, 100);
        ui->slider_param_2->setRange(0, 100);
        ui->slider_param_1->setValue(saltProb);
        ui->slider_param_2->setValue(pepperProb);
        noisyImage = addSaltPepperNoise(originalImage, saltProb, pepperProb);*/
        
    } else { // None 
        noisyImage = cv::Mat();
        ui->Widget_Output_1->clear();
        ui->Label_output_1->setText("Output 1");
        ui->Label_output_2->setText("Output 2");
    }
    
    showImage(noisyImage, ui->Widget_Output_1);
    
    // REAPPLY FILTER
    if (!comingFromFilter) {
        int idx = ui->Combox_Filter->currentIndex();
        onFilterComboBoxSelectionChanged(idx);
    }
}

void Image_Processing::onSliderValueChanged(int value) {
    QObject* senderObj = sender();
    QString noiseType = ui->Combox_Noise->currentText();
    /*qDebug() << "Noise: " << noiseType;*/

    if (senderObj == ui->slider_param_1) {
        qDebug() << "Slider 1 value updated:" << value;
        noiseParam_1 = value;
    }
    else if (senderObj == ui->slider_param_2) {
        qDebug() << "Slider 2 value updated:" << value;
        noiseParam_2 = value;
    }

    if (noiseType == "Gaussian")
        noisyImage = addGaussianNoise(originalImage, noiseParam_1, noiseParam_2);

    else if (noiseType == "Uniform")
        noisyImage = addUniformNoise(originalImage, noiseParam_1, noiseParam_2);

    else if (noiseType == "Salt and Pepper")
        noisyImage = addSaltPepperNoise(originalImage, noiseParam_1, noiseParam_2);
    //else { // None 
    //    noisyImage = cv::Mat();
    //    ui->Widget_Output_1->clear();
    //}
    
    showImage(noisyImage, ui->Widget_Output_1);
    int idx = ui->Combox_Filter->currentIndex();
    onFilterComboBoxSelectionChanged(idx);
}

//NOISE FUNCTIONS

cv::Mat Image_Processing::addSaltPepperNoise(const cv::Mat& img, double saltProb, double pepperProb) {
    cv::Mat noisyImage = img.clone();
    int totalPixels = noisyImage.rows * noisyImage.cols;
    saltProb /= 100;
    pepperProb /= 100;

    // Ensure saltProb and pepperProb are within bounds
    saltProb = clamp(saltProb, 0.0, 1.0);
    pepperProb = clamp(pepperProb, 0.0, 1.0);

    // Salt (white pixels)
    int numSalt = static_cast<int>(totalPixels * saltProb);
    for (int i = 0; i < numSalt; i++) {
        int x = rand() % noisyImage.cols;
        int y = rand() % noisyImage.rows;
        if (noisyImage.channels() == 1) {
            noisyImage.at<uchar>(y, x) = 255;  // White pixel for grayscale
        }
        else {
            noisyImage.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255); // White pixel for color image
        }
    }

    // Pepper (black pixels)
    int numPepper = static_cast<int>(totalPixels * pepperProb);
    for (int i = 0; i < numPepper; i++) {
        int x = rand() % noisyImage.cols;
        int y = rand() % noisyImage.rows;
        if (noisyImage.channels() == 1) {
            noisyImage.at<uchar>(y, x) = 0;  // Black pixel for grayscale
        }
        else {
            noisyImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // Black pixel for color image
        }
    }

    return noisyImage;
}

cv::Mat Image_Processing::addUniformNoise(const cv::Mat& img, double lowerBound, double upperBound) {
    cv::Mat noisyImage = img.clone();

    // Generate uniform noise in the range [-lowerBound, +upperBound]
    cv::Mat noise = cv::Mat::zeros(img.size(), img.type());
    /*cv::randu(noise, -lowerBound, upperBound);*/
    cv::randu(noise, lowerBound, upperBound);

    // Convert noise to the correct format
    noisyImage.convertTo(noisyImage, CV_32F);
    noise.convertTo(noise, CV_32F);

    // Add noise and clip values to valid image range
    noisyImage = noisyImage + noise;
    cv::threshold(noisyImage, noisyImage, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(noisyImage, noisyImage, 0, 0, cv::THRESH_TOZERO);

    // Convert back to original image type
    noisyImage.convertTo(noisyImage, img.type());

    return noisyImage;
}

cv::Mat Image_Processing::addGaussianNoise(const cv::Mat& img, double mean, double standardDev) {

    if (img.empty()) {
        qDebug() << "Can't add gaussian noise. Image is empty!";
        return cv::Mat(); // Return an empty matrix
    }

    cv::Mat tempImage = img.clone();


    // Convert image to float for proper noise addition
    cv::Mat imgFloat;
    if (tempImage.channels() == 3) {
        tempImage.convertTo(imgFloat, CV_32FC3);  // Convert to 3-channel float
    }
    else {
        tempImage.convertTo(imgFloat, CV_32F);  // Convert to 1-channel float
    }


    // Create Gaussian noise matrix with float type
    cv::Mat noise(tempImage.size(), imgFloat.type());
    standardDev /= 2;
    cv::randn(noise, mean, standardDev);  // Mean = 0, StdDev = 30
    /*try {*/

        // Add noise
        noisyImage = imgFloat + noise;
    /*}*/
    //catch (const cv::Exception& e) {  // Catch OpenCV-specific exceptions
    //    /*qDebug() << "ERROR HERE 1:" << e.what();*/
    //    std::cerr << "OpenCV Error: " << e.what() << std::endl;
    //}
    //catch (const std::exception& e) {  // Catch standard C++ exceptions
    //    qDebug() << "ERROR HERE 2:" << e.what();
    //    /*std::cerr << "Standard Exception: " << e.what() << std::endl;*/
    //}


    // Clip pixel values between 0 and 255
        cv::min(noisyImage, 255, noisyImage);
        cv::max(noisyImage, 0, noisyImage);

    // Convert back to 8-bit grayscale
    noisyImage.convertTo(noisyImage, CV_8UC1);
    /*ui->Widget_Output_1->updateImage(noisyImageClipped);*/

    qDebug() << "Gaussian noise added!";

    return noisyImage;
}

// FILTERING
void Image_Processing::onFilterComboBoxSelectionChanged(int index) {
    qDebug() << "APPLYNG FILTER...";
    if (originalImage.empty()) {
        qDebug() << "Cannot apply filter. No image is uploaded.";
        return;
    }

    QString selectedFilter = ui->Combox_Filter->itemText(index);
    qDebug() << "Selected Filter:" << selectedFilter;

    if (selectedFilter == "Average") {
        filteredImage = applyAverageFilter(noisyImage, 3);

    }
    else if (selectedFilter == "Gaussian") {
        filteredImage = applyGaussianFilter(noisyImage, 3);

    }
    else if (selectedFilter == "Median") { // Median
        filteredImage = applyMedianFilter(noisyImage, 3);

    }
    else { // None
        filteredImage = cv::Mat();
        ui->Widget_Output_2->clear();
    }
    showImage(filteredImage, ui->Widget_Output_2);

    // Reapply noise (needed if we were detecting edges before applying the filter)
    int idx = ui->Combox_Noise->currentIndex();
    onNoiseComboBoxSelectionChanged(idx, true);
}

// FILTER FUNCTIONS

cv::Mat Image_Processing::applyAverageFilter(const cv::Mat& img, int kernelSize) {
    cv::Mat filteredImage = img.clone();
    // Apply 3x3 average box filter (excluding borders)
    for (int i = 1; i < img.rows - 1; ++i) {
        for (int j = 1; j < img.cols - 1; ++j) {
            int sum = 0;
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    sum += img.at<uchar>(i + x, j + y);
                }
            }
            filteredImage.at<uchar>(i, j) = sum / 9;
        }
    }

    qDebug() << "Manual Average Filter applied!";

    return filteredImage;

}

cv::Mat Image_Processing::applyGaussianFilter(const cv::Mat& img, int kernelSize) {
    cv::Mat filteredImage = img.clone();

    double gaussianKernel[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
    };

    double kernelSum = 16.0;  // Sum of all kernel elements for normalization

    // Apply 3x3 Gaussian filter manually
    for (int i = 1; i < img.rows - 1; ++i) {
        for (int j = 1; j < img.cols - 1; ++j) {
            double sum = 0.0;

            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    sum += img.at<uchar>(i + x, j + y) * gaussianKernel[x + 1][y + 1];
                }
            }

            filteredImage.at<uchar>(i, j) = static_cast<uchar>(sum / kernelSum);
        }
    }

    qDebug() << "Manual Gaussian Filter applied!";

    return filteredImage;
}

cv::Mat Image_Processing::applyMedianFilter(const cv::Mat& img, int kernelSize) {
    cv::Mat filteredImage = img.clone();

    for (int i = 1; i < img.rows - 1; ++i) {
        for (int j = 1; j < img.cols - 1; ++j) {
            std::vector<uchar> neighbors;

            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    neighbors.push_back(img.at<uchar>(i + x, j + y));
                }
            }

            std::sort(neighbors.begin(), neighbors.end());
            filteredImage.at<uchar>(i, j) = neighbors[4];  // Median value (middle of sorted list)
        }
    }
    qDebug() << "Manual Median Filter applied!";
    return filteredImage;
}

// EDGE DETECTION //

void Image_Processing::onEdgeComboBoxSelectionChanged(int index) {
    qDebug() << "APPLYNG EDGE DETECTION...";
    if (originalImage.empty()) {
        qDebug() << "Cannot detect edges. No image is uploaded.";
        return;
    }

    QString selectedEdgeDetector = ui->Combox_Edges->itemText(index);
    qDebug() << "Selected Edge Detector:" << selectedEdgeDetector;

    ui->Label_output_1->setText("Horizontal Edges");
    ui->Label_output_2->setText("Vertical Edges");

    if (selectedEdgeDetector == "Sobel") {
        applyManualSobel();

    }
    else if (selectedEdgeDetector == "Prewitt") {
        applyManualPrewitt();
    }
    else if (selectedEdgeDetector == "Roberts") {
        ui->Label_output_1->setText("Primary Diagonal Edges");
        ui->Label_output_2->setText("Secondary Diagonal Edges");
        applyManualRoberts();
    }
    else if (selectedEdgeDetector == "Canny") {
        ui->Label_output_1->setText("Edges");
        ui->Label_output_2->setText("Output 2");
        ui->Widget_Output_2->clear();
        cv::Mat cannyEdges;
        Canny(originalImage, cannyEdges, 100, 200);
        showImage(cannyEdges, ui->Widget_Output_1);

    }
    else { // None
        ui->Widget_Output_1->clear();
        ui->Label_output_1->setText("Output 1");
        ui->Label_output_2->setText("Output 2");
    }

}

// EDGE DETECTOR FUNCTIONS //
void Image_Processing::applyManualSobel() {

    if (originalImage.empty()) {
        qDebug() << "Error: Image is empty!";
        return;
    }

    // Create matrices for Sobel X and Y outputs
    cv::Mat sobelX = cv::Mat::zeros(originalImage.size(), CV_64F);
    cv::Mat sobelY = cv::Mat::zeros(originalImage.size(), CV_64F);

    // Manually defined Sobel kernels
    int sobelXKernel[3][3] = { {-1,  0,  1},
                               {-2,  0,  2},
                               {-1,  0,  1} };

    int sobelYKernel[3][3] = { {-1, -2, -1},
                               { 0,  0,  0},
                               { 1,  2,  1} };

    // Apply manual Sobel filtering
    for (int i = 1; i < originalImage.rows - 1; ++i) {
        for (int j = 1; j < originalImage.cols - 1; ++j) {
            double sumX = 0.0, sumY = 0.0;

            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    sumX += originalImage.at<uchar>(i + x, j + y) * sobelXKernel[x + 1][y + 1];
                    sumY += originalImage.at<uchar>(i + x, j + y) * sobelYKernel[x + 1][y + 1];
                }
            }

            sobelX.at<double>(i, j) = sumX;
            sobelY.at<double>(i, j) = sumY;
        }
    }

    // Convert results to displayable format
    cv::Mat sobelX_display, sobelY_display;
    cv::convertScaleAbs(sobelX, sobelX_display);
    cv::convertScaleAbs(sobelY, sobelY_display);

    // Update your promoted widgets with the results
    showImage(sobelX_display, ui->Widget_Output_1);
    showImage(sobelY_display, ui->Widget_Output_2);

    qDebug() << "Manual Sobel Filter applied!";
}

void Image_Processing::applyManualPrewitt() {

    if (originalImage.empty()) {
        qDebug() << "Error: Image is empty!";
        return;
    }
    cv::Mat prewittX = cv::Mat::zeros(originalImage.size(), CV_64F);
    cv::Mat prewittY = cv::Mat::zeros(originalImage.size(), CV_64F);
    int prewittXkernel[3][3] = { {1,0,-1}
                                ,{1,0,-1}
                                ,{1,0,-1} };
    int prewittYkernel[3][3] = { {1,1,1} , {0,0,0} , {-1,-1,-1} };
    for (int i = 1; i < originalImage.rows - 1; ++i) {
        for (int j = 1; j < originalImage.cols - 1; ++j) {
            double sumX = 0.0, sumY = 0.0;

            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; y++) {
                    sumX += originalImage.at<uchar>(i + x, j + y) * prewittXkernel[x + 1][y + 1];
                    sumY += originalImage.at<uchar>(i + x, j + y) * prewittYkernel[x + 1][y + 1];
                }

            }
            prewittX.at<double>(i, j) = sumX;
            prewittY.at<double>(i, j) = sumY;

        }

    }
    cv::Mat prewittX_display, prewittY_display;
    cv::convertScaleAbs(prewittX, prewittX_display);
    cv::convertScaleAbs(prewittY, prewittY_display);
    showImage(prewittX_display, ui->Widget_Output_1);
    showImage(prewittY_display, ui->Widget_Output_2);
    qDebug() << "Manual Prewitt Filter applied!";

}

void Image_Processing::applyManualRoberts() {
    // Primary Diagonal Edges
    double robertsKernel1[2][2] = {
        {1, 0},
        {0, -1},
    };

    cv::Mat output1 = cv::Mat::zeros(originalImage.rows - 1, originalImage.cols - 1, CV_8UC1);

    // Secondary Diagonal Edges
    double robertsKernel2[2][2] = {
        {0, 1},
        {-1, 0}
    };
    cv::Mat output2 = cv::Mat::zeros(originalImage.rows - 1, originalImage.cols - 1, CV_8UC1);


    for (int i = 0; i < originalImage.rows - 1; i++) {
        for (int j = 0; j < originalImage.cols - 1; j++) {
            double region[2][2] = {
                {originalImage.at<uchar>(i, j), originalImage.at<uchar>(i, j + 1)},
                {originalImage.at<uchar>(i + 1, j), originalImage.at<uchar>(i + 1, j + 1)}
            };

            // Primary Diagonal
            double gx = (region[0][0] * robertsKernel1[0][0]) + (region[0][1] * robertsKernel1[0][1]) +
                (region[1][0] * robertsKernel1[1][0]) + (region[1][1] * robertsKernel1[1][1]);
            // Storing Computed Value
            output1.at<uchar>(i, j) = cv::saturate_cast<uchar>(std::abs(gx));

            double gy = (region[0][0] * robertsKernel2[0][0]) + (region[0][1] * robertsKernel2[0][1]) +
                (region[1][0] * robertsKernel2[1][0]) + (region[1][1] * robertsKernel2[1][1]);
            // Storing Computed Value
            output2.at<uchar>(i, j) = cv::saturate_cast<uchar>(std::abs(gy));
        }
    }
    showImage(output1, ui->Widget_Output_1);
    showImage(output2, ui->Widget_Output_2);
}

void Image_Processing::onThresholdingSelected() {

    // Ensure the image is not empty
    if (originalImage.empty()) {
        qDebug() << "Error: No image loaded!";
        return;
    }

    // Prepare Mat objects for output images
    cv::Mat globalThresholded, localThresholded;

    // Apply thresholding
    applyGlobalThresholding(originalImage, globalThresholded);
    applyLocalThresholding(originalImage, localThresholded);

    // Display results
    ui->Label_output_1->setText("Local Thresold");
    ui->Label_output_2->setText("Global Thresold");
    showImage(localThresholded, ui->Widget_Output_1);
    showImage(globalThresholded, ui->Widget_Output_2);
}

// Apply Global Thresholding (Otsu's Method)
void Image_Processing::applyGlobalThresholding(const cv::Mat& img, cv::Mat& output) {
    cv::threshold(img, output, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    qDebug() << "Global thresholding applied (Otsu's Method)";
}

// Apply Local Adaptive Thresholding
void Image_Processing::applyLocalThresholding(const cv::Mat& img, cv::Mat& output) {
    cv::adaptiveThreshold(img, output, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    qDebug() << "Local thresholding applied (Adaptive Gaussian)";
}

// HISTOGRAM AND DISTRIBUTION //
void Image_Processing::drawHistogramAndDistribution() {
    if (ui->RadioButton_Histogram->isChecked()) {
        ui->Label_output_1->setText("Histogram");
        ui->Label_output_2->setText("CDF Distribution");
        if (ui->checkBox_grayImg->isChecked()) {
            qDebug() << "Applying gray histogram...";
            grayHistogramAndDistribution();
        }
        else {
            qDebug() << "Applying rgb histogram...";
            rgbHistogramAndDistribution();
        }
    }
    else {
        qDebug() << "Please check Histogram and Distribution first.";
    }
    
}

//void Image_Processing::onHistogramCheckBoxStateChanged(int state) {
//    if (state == Qt::Checked) {
//
//    }
//    else {
//
//    }
//}

void Image_Processing::rgbHistogramAndDistribution() {
    showImage(originalImageColored, ui->Widget_Org_Image);
    cv::Mat img = originalImageColored.clone();

    if (img.empty() || img.channels() != 3) {
        qDebug() << "Error: Invalid RGB image.";
        return;
    }

    // Split image into R, G, B channels
    std::vector<cv::Mat> bgrChannels;
    cv::split(img, bgrChannels);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    // Compute histograms
    cv::Mat histR, histG, histB;
    cv::calcHist(&bgrChannels[2], 1, 0, cv::Mat(), histR, 1, &histSize, &histRange, uniform, accumulate); // Red
    cv::calcHist(&bgrChannels[1], 1, 0, cv::Mat(), histG, 1, &histSize, &histRange, uniform, accumulate); // Green
    cv::calcHist(&bgrChannels[0], 1, 0, cv::Mat(), histB, 1, &histSize, &histRange, uniform, accumulate); // Blue

    // Compute CDFs
    cv::Mat cdfR = histR.clone(), cdfG = histG.clone(), cdfB = histB.clone();
    for (int i = 1; i < histSize; i++) {
        cdfR.at<float>(i) += cdfR.at<float>(i - 1);
        cdfG.at<float>(i) += cdfG.at<float>(i - 1);
        cdfB.at<float>(i) += cdfB.at<float>(i - 1);
    }

    cdfR /= cdfR.at<float>(histSize - 1); // Normalize to [0,1]
    cdfG /= cdfG.at<float>(histSize - 1);
    cdfB /= cdfB.at<float>(histSize - 1);

    // Image settings
    int histW = 512, histH = 150;
    int binW = cvRound((double)histW / histSize);

    // Create blank images for histograms and CDFs (3 stacked)
    cv::Mat histStack(histH * 3, histW, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat cdfStack(histH * 3, histW, CV_8UC3, cv::Scalar(255, 255, 255));

    // Normalize histogram values
    cv::normalize(histR, histR, 0, histH, cv::NORM_MINMAX);
    cv::normalize(histG, histG, 0, histH, cv::NORM_MINMAX);
    cv::normalize(histB, histB, 0, histH, cv::NORM_MINMAX);

    // Function to draw histogram or CDF graph
    auto drawGraph = [&](cv::Mat& img, cv::Mat& data, cv::Scalar color, int offsetY) {
        for (int i = 0; i < histSize; i++) {
            cv::line(img,
                cv::Point(binW * i, offsetY + histH),
                cv::Point(binW * i, offsetY + histH - cvRound(data.at<float>(i))),
                color, 2);
        }
        };

    // Draw stacked histograms
    drawGraph(histStack, histB, cv::Scalar(255, 0, 0), 0);        // Blue
    drawGraph(histStack, histG, cv::Scalar(0, 255, 0), histH);     // Green
    drawGraph(histStack, histR, cv::Scalar(0, 0, 255), histH * 2); // Red

    // Draw stacked CDFs
    for (int i = 1; i < histSize; i++) {
        cv::line(cdfStack,
            cv::Point(binW * (i - 1), histH - cvRound(cdfB.at<float>(i - 1) * histH)),
            cv::Point(binW * i, histH - cvRound(cdfB.at<float>(i) * histH)),
            cv::Scalar(255, 0, 0), 2); // Blue

        cv::line(cdfStack,
            cv::Point(binW * (i - 1), histH * 2 - cvRound(cdfG.at<float>(i - 1) * histH)),
            cv::Point(binW * i, histH * 2 - cvRound(cdfG.at<float>(i) * histH)),
            cv::Scalar(0, 255, 0), 2); // Green

        cv::line(cdfStack,
            cv::Point(binW * (i - 1), histH * 3 - cvRound(cdfR.at<float>(i - 1) * histH)),
            cv::Point(binW * i, histH * 3 - cvRound(cdfR.at<float>(i) * histH)),
            cv::Scalar(0, 0, 255), 2); // Red
    }

    // Draw axes
    auto drawAxes = [&](cv::Mat& img) {
        cv::line(img, cv::Point(0, histH), cv::Point(histW, histH), cv::Scalar(0, 0, 0), 2);
        cv::line(img, cv::Point(0, histH * 2), cv::Point(histW, histH * 2), cv::Scalar(0, 0, 0), 2);
        cv::line(img, cv::Point(0, histH * 3), cv::Point(histW, histH * 3), cv::Scalar(0, 0, 0), 2);
        };

    drawAxes(histStack);
    drawAxes(cdfStack);

    // Show results
    showImage(histStack, ui->Widget_Output_1); // RGB Histograms
    showImage(cdfStack, ui->Widget_Output_2); // RGB CDFs

    qDebug() << "RGB Histograms and CDFs drawn successfully.";
}


void Image_Processing::grayHistogramAndDistribution() {
    showImage(originalImage, ui->Widget_Org_Image);

    // Ensure valid grayscale image (Not needed I already make sure it's gray upon upload)
    /*if (grayImage.empty() || grayImage.channels() != 1) {
        qDebug() << "Error: Invalid grayscale image.";
        return;
    } */

    // Histogram settings
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    // Compute histogram
    cv::Mat hist;
    cv::calcHist(&originalImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    // Compute CDF
    cv::Mat cdf = hist.clone();
    for (int i = 1; i < histSize; i++) {
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    }
    cdf /= cdf.at<float>(histSize - 1); // Normalize CDF to [0,1]

    // Histogram & CDF Display Settings
    int histW = 512, histH = 150;
    int binW = cvRound((double)histW / histSize);

    cv::Mat histImage(histH, histW, CV_8UC3, cv::Scalar(255, 255, 255)); // Histogram background
    cv::Mat cdfImage(histH, histW, CV_8UC3, cv::Scalar(255, 255, 255)); // CDF background

    // Normalize histogram values for display
    cv::normalize(hist, hist, 0, histH, cv::NORM_MINMAX);

    // Draw Histogram Bars
    for (int i = 0; i < histSize; i++) {
        cv::rectangle(histImage,
            cv::Point(binW * i, histH),
            cv::Point(binW * (i + 1), histH - cvRound(hist.at<float>(i))),
            cv::Scalar(0, 0, 0),
            cv::FILLED);
    }

    // Draw CDF Curve
    for (int i = 1; i < histSize; i++) {
        cv::line(cdfImage,
            cv::Point(binW * (i - 1), histH - cvRound(cdf.at<float>(i - 1) * histH)),
            cv::Point(binW * i, histH - cvRound(cdf.at<float>(i) * histH)),
            cv::Scalar(0, 0, 0), 2);
    }

    // Draw Axes
    cv::line(histImage, cv::Point(0, histH), cv::Point(histW, histH), cv::Scalar(0, 0, 0), 2); // X-axis
    cv::line(histImage, cv::Point(0, 0), cv::Point(0, histH), cv::Scalar(0, 0, 0), 2); // Y-axis

    cv::line(cdfImage, cv::Point(0, histH), cv::Point(histW, histH), cv::Scalar(0, 0, 0), 2); // X-axis
    cv::line(cdfImage, cv::Point(0, 0), cv::Point(0, histH), cv::Scalar(0, 0, 0), 2); // Y-axis

    // Display in UI
    showImage(histImage, ui->Widget_Output_1); // Histogram
    showImage(cdfImage, ui->Widget_Output_2); // CDF

    qDebug() << "Grayscale Histogram and CDF drawn successfully.";
}

void Image_Processing::normalizeAndEqualize() {
    showImage(originalImage, ui->Widget_Org_Image);

    ui->Label_output_1->setText("Normalized Image");
    ui->Label_output_2->setText("Equalized Image");

    // Normalization
    cv::Mat normalizedImage = originalImage.clone();
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(originalImage, &minVal, &maxVal, &minLoc, &maxLoc);

    for (int i = 0; i < originalImage.rows; i++) {
        for (int j = 0; j < originalImage.cols; j++) {
            normalizedImage.at<uchar>(i, j) = static_cast<uchar>(255 *
                (originalImage.at<uchar>(i, j) - minVal) / (maxVal - minVal));
        }
    }
    showImage(normalizedImage, ui->Widget_Output_1);

    // Equalization
    cv::Mat equalizedImage;
    cv::equalizeHist(originalImage, equalizedImage);
    showImage(equalizedImage, ui->Widget_Output_2);
}