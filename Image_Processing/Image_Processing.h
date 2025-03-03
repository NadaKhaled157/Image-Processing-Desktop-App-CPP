#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <QMainWindow>
#include <QFileDialog>
#include <QEvent>
#include <QMouseEvent>
#include <QLabel>
#include "ui_image_processing.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

namespace Ui {
    class MainWindow;
}

class Image_Processing : public QMainWindow {
    Q_OBJECT

public:
    explicit Image_Processing(QWidget* parent = nullptr);
    ~Image_Processing();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private slots:
     void onNoiseComboBoxSelectionChanged(int index, bool comingFromFilter);
     void onFilterComboBoxSelectionChanged(int index);
     void onEdgeComboBoxSelectionChanged(int index);
     void onSliderValueChanged(int value); // Noise Slider
     void onKernelSizeSliderValueChanged(int value);
     void showImage(const cv::Mat& img, QLabel* imageLabel);

     // Noise
     cv::Mat addSaltPepperNoise(const cv::Mat& img, double saltProb, double pepperProb);
     cv::Mat addUniformNoise(const cv::Mat& img, double lowerBound, double upperBound);
     cv::Mat addGaussianNoise(const cv::Mat& img, double mean, double standard_dev);

     // Filters

     cv::Mat applyAverageFilter(const cv::Mat& img, int kernelSize);
     cv::Mat applyGaussianFilter(const cv::Mat& img, int kernelSize);
     cv::Mat applyMedianFilter(const cv::Mat& img, int kernelSize);

     // Edge Detection

     void applyManualSobel();
     void applyManualPrewitt();
     void applyManualRoberts();

     // Histogram and Distribution

     void drawHistogramAndDistribution(); // Check whether it's RGB or Gray scale
     void rgbHistogramAndDistribution();
     void grayHistogramAndDistribution();
     /*void onHistogramCheckBoxStateChanged(int state);*/

     // Normalize and Equalize

     void normalizeAndEqualize();

     // Thresholding

     void onThresholdingSelected();
     void applyGlobalThresholding(const cv::Mat& img, cv::Mat& output);
     void applyLocalThresholding(const cv::Mat& img, cv::Mat& output);

     // Frequency Domain Filters

     void applyFrequencyDomainMixer();
     void applyFrequencyDomainFilter();
     cv::Mat applyFrequencyFilter(const cv::Mat& inputImage, int radius, bool isLowPass);
     cv::Mat createCircularMask(cv::Size size, int radius, bool isLowPass);
     void shiftDFT(cv::Mat& magImage);
    

private:
    Ui::MainWindow* ui;
    QPixmap imagePixmap; // to store the uploaded image
    cv::Mat originalImageColored, originalImage, secondImage, noisyImage, filteredImage;
    double noiseParam_1 = 50; // Uniform: lowerBound, Gaussian: - , Salt and Pepper: Salt
    double noiseParam_2 = 50; // Uniform: upperBound, Gaussian: - , Salt and Pepper: Pepper
    int kernelSize = 1;
    bool detectingEdges;
};

#endif // IMAGE_PROCESSING_H