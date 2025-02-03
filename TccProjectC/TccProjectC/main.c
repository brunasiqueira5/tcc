#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <stdio.h>
#include "functions.h"

void sendImages(char* imgPath, const char* resultWindow, void* svm) {
    int contaImg = 1;

    while (1) {
        snprintf(imgPath, 256, "C:\\Dev\\tcc_project_c\\image\\imagem%d.jpg", contaImg);
        IplImage* img = cvLoadImage(imgPath, CV_LOAD_IMAGE_COLOR);
        if (img == NULL) break;

        IplImage* imgGray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
        cvCvtColor(img, imgGray, CV_BGR2GRAY);

        IplImage* imgThreshold = cvCreateImage(cvGetSize(imgGray), IPL_DEPTH_8U, 1);
        cvAdaptiveThreshold(imgGray, imgThreshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, 1);

        IplImage* imgDil = cvCreateImage(cvGetSize(imgThreshold), IPL_DEPTH_8U, 1);
        cvDilate(imgThreshold, imgDil, NULL, 1);

        // Detectar e classificar os tardígrados
        detectTardigrades(svm, imgDil, img);

        showAndWaitResized(resultWindow, img, img->width / 2, img->height / 2);

        cvReleaseImage(&img);
        cvReleaseImage(&imgGray);
        cvReleaseImage(&imgThreshold);
        cvReleaseImage(&imgDil);
        contaImg++;
    }
}

int main() {
    char imgPath[256];
    const char* resultWindow = "Resultado";

    // Criar e treinar o SVM, Support Vector Machine, é um algoritmo de aprendizado de máquina supervisionado que classifica dados. 
    void* svm = createSVM();
    trainSVM(svm);

    sendImages(imgPath, resultWindow, svm);

    cvDestroyAllWindows();
    return 0;
}