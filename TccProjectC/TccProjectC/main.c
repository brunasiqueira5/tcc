#include <opencv2/core/core_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

#include <stdio.h>

// Fun��o para redimensionar e exibir imagens
void showAndWaitResized(const char* windowName, IplImage* img, int width, int height) {
    // Criar uma imagem de destino com o tamanho especificado
    IplImage* resizedImg = cvCreateImage(cvSize(width, height), img->depth, img->nChannels);
    // Redimensionar a imagem usando interpolacao linear
    cvResize(img, resizedImg, CV_INTER_LINEAR);

    // Exibir a imagem redimensionada
    cvShowImage(windowName, resizedImg);
    cvWaitKey(0);

    // Liberar a mem�ria da imagem redimensionada
    cvReleaseImage(&resizedImg);
};

// Fun��o para processar contornos
char getContours(IplImage* imgDil, IplImage* img) {
    // Criar armazenamento de mem�ria para os contornos
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contours = NULL;

    // Encontrar contornos na imagem dilatada
    int num_contours = cvFindContours(
        imgDil, storage, &contours, sizeof(CvContour),
        CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0)
    );

    printf("Numero de contornos encontrados: %d\n", num_contours);

    // Iterar sobre os contornos encontrados
    for (CvSeq* current = contours; current != NULL; current = current->h_next) {
        double area = cvContourArea(current, CV_WHOLE_SEQ, 0);

        // Desenhar os contornos apenas se a �rea for maior que um valor m�nimo
        if (area > 1000) {
            float perimetro = cvArcLength(current, CV_WHOLE_SEQ, 0);

            // Criar armazenamento para aproxima��o poligonal
            CvMemStorage* storage_poly = cvCreateMemStorage(0);

            // Aproxima��o poligonal do contorno
            CvSeq* approx = cvApproxPoly(
                current, sizeof(CvContour), storage_poly,
                CV_POLY_APPROX_DP, 0.0005 * perimetro, 1
            );

            // Desenhar os contornos aproximados na imagem original
            cvDrawContours(img, approx, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), 0, 2, 8, cvPoint(0, 0));

            // Liberar mem�ria do armazenamento
            cvReleaseMemStorage(&storage_poly);
        }
    }

    // Liberar mem�ria do armazenamento principal
    cvReleaseMemStorage(&storage);
}

int main() {
    char imgPath[256];
    int contaImg = 1; // Contador de imagens
    const char* grayWindow = "Imagem em Escala de Cinza";
    const char* dilateWindow = "Imagem Dilatada";

    while (1) {
        // Gerar o caminho da imagem
        snprintf(imgPath, sizeof(imgPath), "C:\\Dev\\tcc_project_c\\image\\imagem%d.jpg", contaImg);

        // Carregar a imagem original
        IplImage* img = cvLoadImage(imgPath, CV_LOAD_IMAGE_COLOR);
        if (img == NULL) {
            break; // Parar se n�o houver mais imagens
        }

        // Criar um kernel estruturante para opera��es morfol�gicas
        IplConvKernel* kernel = cvCreateStructuringElementEx(5, 5, 2, 2, CV_SHAPE_RECT, NULL);

        // Aplicar um desfoque para suavizar ru�dos
        IplImage* imgKernel = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, img->nChannels);
        cvSmooth(img, imgKernel, CV_GAUSSIAN, 9, 9, 9, 9);
        cvWaitKey(0);

        // Remover ru�dos externos com opera��o de abertura morfol�gica
        IplImage* imgMorphoOpen = cvCreateImage(cvGetSize(imgKernel), IPL_DEPTH_8U, imgKernel->nChannels);
        cvMorphologyEx(imgKernel, imgMorphoOpen, NULL, kernel, CV_MOP_OPEN, 1);

        // Remover ru�dos internos com opera��o de fechamento morfol�gico
        IplImage* imgMorphoClose = cvCloneImage(imgMorphoOpen);
        cvMorphologyEx(imgMorphoOpen, imgMorphoClose, NULL, kernel, CV_MOP_CLOSE, 1);

        // Converter imagem para escala de cinza
        IplImage* imgGray = cvCreateImage(cvGetSize(imgMorphoClose), IPL_DEPTH_8U, 1);
        cvCvtColor(imgMorphoClose, imgGray, CV_BGR2GRAY);

        // Aplicar limiariza��o adaptativa para binarizar a imagem
        IplImage* imgThreshold = cvCreateImage(cvGetSize(imgGray), IPL_DEPTH_8U, 1);
        cvAdaptiveThreshold(imgGray, imgThreshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, 1);

        // Aplicar dilata��o para melhorar a detec��o de contornos
        IplImage* imgDil = cvCreateImage(cvGetSize(imgThreshold), IPL_DEPTH_8U, 1);
        cvDilate(imgThreshold, imgDil, NULL, 1);

        // Processar os contornos na imagem binarizada
        getContours(imgDil, img);

        // Exibir imagem redimensionada para verifica��o
        showAndWaitResized(dilateWindow, img, img->width / 2, img->height / 2);

        // Liberar mem�ria alocada para imagens processadas
        cvReleaseImage(&img);
        cvReleaseImage(&imgMorphoOpen);
        cvReleaseImage(&imgMorphoClose);
        cvReleaseImage(&imgGray);
        cvReleaseImage(&imgKernel);
        cvReleaseImage(&imgThreshold);
        cvReleaseImage(&imgDil);

        // Incrementar o contador para processar a pr�xima imagem
        contaImg++;
    }

    // Fechar todas as janelas do OpenCV
    cvDestroyAllWindows();

    return 0;
}