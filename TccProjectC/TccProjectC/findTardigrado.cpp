#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>

extern "C" {
#include "functions.h"
}

using namespace cv;
using namespace std;

void showAndWaitResized(const char* windowName, IplImage* img, int width, int height) {
    IplImage* resizedImg = cvCreateImage(cvSize(width, height), img->depth, img->nChannels);
    cvResize(img, resizedImg, CV_INTER_LINEAR);
    cvShowImage(windowName, resizedImg);
    cvWaitKey(0);
    cvReleaseImage(&resizedImg);
}

Mat computeHOG(IplImage* img) { //objetivo extrair informacoes referentes tes a orientacao das arestas existentes em uma imagem
    Mat matImg = cvarrToMat(img); //Converte a imagem IplImage* img
    vector<float> descriptors;
    HOGDescriptor hog( // Cria um objeto HOGDescriptor com par�metros espec�ficos
        Size(64, 64), Size(16, 16),
        Size(8, 8), Size(8, 8), 9
    );
    resize(matImg, matImg, Size(64, 64)); 
    hog.compute(matImg, descriptors); //Calcula o descritor HOG da imagem matImg e armazena o resultado no vetor descriptors
    return Mat(descriptors).reshape(1, 1); //Converte o vetor descriptors (que � um std::vector<float>) para uma matriz do OpenCV (Mat)
}

void* createSVM() {
    return new SVM();
}

void trainSVM(void* svm) {
    Ptr<SVM> svmPtr = *static_cast<Ptr<SVM>*>(svm); //converte um ponteiro gen�rico void* svm para um ponteiro inteligente Ptr<SVM>

    vector<Mat> trainingData; //armazena as caracter�sticas dessas amostras
    vector<int> labels; //fornece a informa��o sobre a qual classe cada amostra de treinamento pertence

    // Carregar imagens de treino
    const char* tardigradeImages[] = { "C:\\Dev\\tcc_project_c\\trainingImages\\tardigrado1.jpg", "C:\\Dev\\tcc_project_c\\trainingImages\\tardigrado2.jpg" };
    const char* nonTardigradeImages[] = { "C:\\Dev\\tcc_project_c\\trainingImages\\naoTartigrado\\naoTardigrado1.jpeg", "C:\\Dev\\tcc_project_c\\trainingImages\\naoTartigrado\\naoTardigrado2.jpeg" };

    // Processar imagens de tard�grados (classe 1)
    for (int i = 0; i < 2; i++) {
        IplImage* img = cvLoadImage(tardigradeImages[i], CV_LOAD_IMAGE_GRAYSCALE);
        if (img) {
            trainingData.push_back(computeHOG(img)); //armazena as caracter�sticas da imagem (img), calculadas pela fun��o computeHOG, no vetor trainingData
            labels.push_back(1); //associa essas caracter�sticas ao r�tulo (label) 1, indicando que essa imagem pertence � classe positiva (no seu caso, tard�grados).
            cvReleaseImage(&img);
        }
    }

    // Processar imagens de fundo (classe 0)
    for (int i = 0; i < 2; i++) {
        IplImage* img = cvLoadImage(nonTardigradeImages[i], CV_LOAD_IMAGE_GRAYSCALE);
        if (img) {
            trainingData.push_back(computeHOG(img));
            labels.push_back(0);
            cvReleaseImage(&img);
        }
    }

    // Converter para Mat
    Mat trainingDataMat, labelsMat; //Declara duas matrizes do OpenCV (Mat). trainingDataMat armazenar� os dados de treinamento (as caracter�sticas HOG), e labelsMat armazenar� os r�tulos correspondentes.
    vconcat(trainingData, trainingDataMat); //concatena verticalmente os vetores de caracter�sticas armazenados em trainingData (que � um std::vector<Mat>) em uma �nica matriz trainingDataMat
    labelsMat = Mat(labels).reshape(1, labels.size()); //Cria uma matriz labelsMat a partir do vetor labels (que � um std::vector<int>). O reshape(1, labels.size()) transforma essa matriz em uma matriz linha.  Isso � necess�rio porque o m�todo train do SVM espera os r�tulos em um formato espec�fico

    // Treinar SVM
    svmPtr->train(trainingDataMat, labelsMat); //realiza o treinamento do SVM usando os dados de treinamento (trainingDataMat) e os r�tulos correspondentes (labelsMat)
}

float predictSVM(void* svm, IplImage* img) {
    Ptr<SVM> svmPtr = *static_cast<Ptr<SVM>*>(svm);
    Mat features = computeHOG(img);
    return svmPtr->predict(features);
}

void detectTardigrades(void* svm, IplImage* imgDil, IplImage* img) {
    Ptr<SVM> svmPtr = *static_cast<Ptr<SVM>*>(svm);  // Converte um ponteiro gen�rico void* svm para um ponteiro inteligente Ptr<SVM>

    if (!svmPtr.empty()) {
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* contours = NULL;

        // Encontrar contornos
        cvFindContours(imgDil, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

        for (CvSeq* current = contours; current != NULL; current = current->h_next) { //Itera sobre cada contorno encontrado
            double area = cvContourArea(current, CV_WHOLE_SEQ, 0);
            if (area > 1000) {
                CvMemStorage* storage_poly = cvCreateMemStorage(0);
                CvSeq* approx = cvApproxPoly(current, sizeof(CvContour), storage_poly, CV_POLY_APPROX_DP, 0.02 * cvArcLength(current, CV_WHOLE_SEQ, 0), 1); //Aproxima o contorno por um pol�gono

                // Criar bounding box
                CvRect boundingBox = cvBoundingRect(approx, 0); //: Calcula a "bounding box" (ret�ngulo envolvente) do pol�gono aproximado
                cvSetImageROI(img, boundingBox);
                IplImage* roiImg = cvCreateImage(cvSize(boundingBox.width, boundingBox.height), img->depth, img->nChannels);
                cvCopy(img, roiImg, NULL);
                cvResetImageROI(img);

                // Converter para escala de cinza
                IplImage* roiGray = cvCreateImage(cvSize(boundingBox.width, boundingBox.height), IPL_DEPTH_8U, 1);
                cvCvtColor(roiImg, roiGray, CV_BGR2GRAY);

                // Extrair caracter�sticas e classificar
                Mat features = computeHOG(roiGray);

                // Corre��o: Usar svmPtr
                float prediction = svmPtr->predict(features);

                // Se for um tard�grado, desenhar em verde, sen�o em vermelho
                CvScalar color = (prediction == 1) ? CV_RGB(0, 255, 0) : CV_RGB(255, 0, 0);
                cvRectangle(img, cvPoint(boundingBox.x, boundingBox.y), cvPoint(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height), color, 2);

                cvReleaseImage(&roiImg);
                cvReleaseImage(&roiGray);
                cvReleaseMemStorage(&storage_poly);
            }
        }

        cvReleaseMemStorage(&storage);
    }
    else {
        std::cerr << "Erro: Ponteiro SVM inv�lido!" << std::endl;
    }

    return;
}
