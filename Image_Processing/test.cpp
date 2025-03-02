//void Image_Processing::applyFrequencyDomainMixer()
//{
//    // Only proceed if the mixer radio button is checked
//    if (!ui->RadioButton_Mixer->isChecked()) {
//        return;
//    }
//
//    // Check if images are empty
//    if (originalImage.empty() || secondImage.empty()) {
//        qDebug() << "Error: One or both input images are empty";
//        return;
//    }
//
//    // Ensure both images have the same size
//    if (originalImage.size() != secondImage.size()) {
//        // Resize second image to match the original if needed
//        cv::resize(secondImage, secondImage, originalImage.size());
//    }
//
//    // Get the radius (cutoff frequency) from each slider
//    int radiusImg1 = ui->slider_freq_img1->value();
//    int radiusImg2 = ui->slider_freq_img2->value();
//
//    // Process originalImage with low-pass filter using slider_freq_img1
//    cv::Mat lowPassResult = applyFrequencyFilter(originalImage, radiusImg1, true);
//
//    // Process secondImage with high-pass filter using slider_freq_img2
//    cv::Mat highPassResult = applyFrequencyFilter(secondImage, radiusImg2, false);
//
//    // Mix the two filtered images
//    cv::Mat mixedResult;
//    cv::addWeighted(lowPassResult, 0.5, highPassResult, 0.5, 0, mixedResult);
//
//    // Update display widget
//    showImage(mixedResult, ui->Widget_Output_2);
//
//    qDebug() << "Frequency domain mixer applied with cutoffs: Image1=" << radiusImg1 << ", Image2=" << radiusImg2;
//}
//
//// Helper function to apply frequency domain filter (low-pass or high-pass)
//cv::Mat Image_Processing::applyFrequencyFilter(const cv::Mat& inputImage, int radius, bool isLowPass)
//{
//    // Get optimal size for DFT
//    int rows = cv::getOptimalDFTSize(inputImage.rows);
//    int cols = cv::getOptimalDFTSize(inputImage.cols);
//
//    // Create padded image for DFT
//    cv::Mat padded;
//    cv::copyMakeBorder(inputImage, padded, 0, rows - inputImage.rows, 0, cols - inputImage.cols,
//        cv::BORDER_CONSTANT, cv::Scalar::all(0));
//
//    // Convert to float for DFT operations
//    cv::Mat paddedFloat;
//    padded.convertTo(paddedFloat, CV_32F);
//
//    // Prepare complex image for DFT result (real + imaginary parts)
//    cv::Mat planes[] = {
//        paddedFloat,
//        cv::Mat::zeros(paddedFloat.size(), CV_32F)
//    };
//    cv::Mat complexImage;
//    cv::merge(planes, 2, complexImage);
//
//    // Perform DFT
//    cv::dft(complexImage, complexImage);
//
//    // Shift zero frequency to center for filtering
//    shiftDFT(complexImage);
//
//    // Create filter mask
//    cv::Mat mask = createCircularMask(complexImage.size(), radius, isLowPass);
//
//    // Apply filter
//    cv::Mat filterResult;
//    cv::mulSpectrums(complexImage, mask, filterResult, 0);
//
//    // Shift back before inverse DFT
//    shiftDFT(filterResult);
//
//    // Perform inverse DFT
//    cv::Mat inverseResult;
//    cv::idft(filterResult, inverseResult, cv::DFT_REAL_OUTPUT);
//
//    // Normalize result for display
//    cv::normalize(inverseResult, inverseResult, 0, 255, cv::NORM_MINMAX);
//
//    // Convert back to 8-bit for display
//    cv::Mat result;
//    inverseResult.convertTo(result, CV_8U);
//
//    return result;
//}
//
//void Image_Processing::applyFrequencyDomainFilter()
//{
//    // Only proceed if the domain filter radio button is checked
//    if (!ui->RadioButton_Domain_Filter->isChecked()) {
//        return;
//    }
//
//    // Get the original grayscale image
//    cv::Mat image = ui->Widget_Org_Image->getImage();
//
//    // Check if image is empty
//    if (image.empty()) {
//        qDebug() << "Error: Input image is empty";
//        return;
//    }
//
//    // Get the radius (cutoff frequency) from slider
//    int radius = ui->slider_freq_domain->value();
//
//    // Apply low-pass filter
//    cv::Mat inverseLowPass = applyFrequencyFilter(image, radius, true);
//
//    // Apply high-pass filter
//    cv::Mat inverseHighPass = applyFrequencyFilter(image, radius, false);
//
//    // Update display widgets
//    ui->Widget_Output_1->updateImage(inverseLowPass);  // Low-pass output
//    ui->Widget_Output_2->updateImage(inverseHighPass); // High-pass output
//
//    qDebug() << "Frequency domain filter applied with cutoff radius: " << radius;
//}
//
//// Helper function to create circular mask (low-pass or high-pass)
//cv::Mat Image_Processing::createCircularMask(cv::Size size, int radius, bool isLowPass)
//{
//    // Create a complex mask (2 channels: real + imaginary)
//    cv::Mat mask(size, CV_32FC2, cv::Scalar(0, 0));
//    cv::Point center(size.width / 2, size.height / 2);
//
//    // Create the circular mask
//    for (int i = 0; i < size.height; i++) {
//        for (int j = 0; j < size.width; j++) {
//            double distance = std::sqrt(pow(i - center.y, 2) + pow(j - center.x, 2));
//
//            if (isLowPass) {
//                // Low-pass mask: 1 inside the circle, 0 outside
//                if (distance <= radius) {
//                    mask.at<cv::Vec2f>(i, j)[0] = 1.0; // Real part
//                    mask.at<cv::Vec2f>(i, j)[1] = 0.0; // Imaginary part
//                }
//            }
//            else {
//                // High-pass mask: 0 inside the circle, 1 outside
//                if (distance > radius) {
//                    mask.at<cv::Vec2f>(i, j)[0] = 1.0; // Real part
//                    mask.at<cv::Vec2f>(i, j)[1] = 0.0; // Imaginary part
//                }
//            }
//        }
//    }
//
//    return mask;
//}
//
//// Helper function to shift zero frequency to/from center
//void Image_Processing::shiftDFT(cv::Mat& magImage)
//{
//    int cx = magImage.cols / 2;
//    int cy = magImage.rows / 2;
//
//    // Create ROI for quadrants
//    cv::Mat q0(magImage, cv::Rect(0, 0, cx, cy));      // Top-left
//    cv::Mat q1(magImage, cv::Rect(cx, 0, cx, cy));     // Top-right
//    cv::Mat q2(magImage, cv::Rect(0, cy, cx, cy));     // Bottom-left
//    cv::Mat q3(magImage, cv::Rect(cx, cy, cx, cy));    // Bottom-right
//
//    // Swap quadrants (Top-left with Bottom-right)
//    cv::Mat tmp;
//    q0.copyTo(tmp);
//    q3.copyTo(q0);
//    tmp.copyTo(q3);
//
//    // Swap quadrants (Top-right with Bottom-left)
//    q1.copyTo(tmp);
//    q2.copyTo(q1);
//    tmp.copyTo(q2);
//}


//////////////////////////////////////////////////////////////
//ui->setupUi(this);
//// Call processImage() after an image is selected
//connect(ui->Widget_Org_Image, &ImageDisplayWidget::imageSelected, this, &imageprocessing::processImage);
//connect(ui->Combox_Noise, &QComboBox::currentTextChanged, this, [this](const QString& noiseType) {
//    double noiseParam_1 = ui->slider_param_1->value();
//    double noiseParam_2 = ui->slider_param_2->value();
//    onNoiseTypeChanged(noiseType, noiseParam_1, noiseParam_2);
//    });
//connect(ui->Combox_Filter, &QComboBox::currentTextChanged, this, &imageprocessing::onFilterTypeChanged);
//connect(ui->slider_param_1, &QSlider::valueChanged, this, [this]() {
//    QString noiseType = ui->Combox_Noise->currentText();
//    double noiseParam_1 = ui->slider_param_1->value();
//    double noiseParam_2 = ui->slider_param_2->value();
//    onNoiseTypeChanged(noiseType, noiseParam_1, noiseParam_2);
//    });
//connect(ui->slider_param_2, &QSlider::valueChanged, this, [this]() {
//    QString noiseType = ui->Combox_Noise->currentText();
//    double noiseParam_1 = ui->slider_param_1->value();
//    double noiseParam_2 = ui->slider_param_2->value();
//    onNoiseTypeChanged(noiseType, noiseParam_1, noiseParam_2);
//    });
//connect(ui->Combox_Edges, &QComboBox::currentTextChanged, this, &imageprocessing::onEdgeTypeChanged);
//connect(ui->RadioButton_Threshold, &QRadioButton::toggled, this, &imageprocessing::onThresholdingSelected);
//connect(ui->RadioButton_Domain_Filter, &QRadioButton::toggled, this, &imageprocessing::applyFrequencyDomainFilter);
//connect(ui->slider_freq_domain, &QSlider::valueChanged, this, &imageprocessing::applyFrequencyDomainFilter);
//
//
//// Connect the mixer radio button
//connect(ui->RadioButton_Mixer, &QRadioButton::toggled, this, &imageprocessing::applyFrequencyDomainMixer);
//
//// Connect the sliders for the mixer
//connect(ui->slider_freq_img1, &QSlider::valueChanged, this, &imageprocessing::applyFrequencyDomainMixer);
//connect(ui->slider_freq_img2, &QSlider::valueChanged, this, &imageprocessing::applyFrequencyDomainMixer);