#include <opencv2/core/core_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

#include <stdio.h>
#include <stdlib.h>

// Função para redimensionar e exibir imagens
void showAndWaitResized(const char* windowName, IplImage* img, int width, int height) {
    // Criar uma imagem de destino com o tamanho especificado
    IplImage* resizedImg = cvCreateImage(cvSize(width, height), img->depth, img->nChannels);
    cvResize(img, resizedImg, CV_INTER_LINEAR); // Redimensionar usando interpolação linear

    // Exibir a imagem redimensionada
    cvShowImage(windowName, resizedImg);
    cvWaitKey(0);

    // Liberar a memória da imagem redimensionada
    cvReleaseImage(&resizedImg);
};

// Função para processar contornos
char getContours(IplImage* imgDil, IplImage* img) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contours = NULL;

    // Encontrar contornos
    int num_contours = cvFindContours(
        imgDil, storage, &contours, sizeof(CvContour),
        CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0)
    );

    printf("Numero de contornos encontrados: %d\n", num_contours);

    // Interar sobre os contornos encontrados
    for (CvSeq* current = contours; current != NULL; current = current->h_next) {
        double area = cvContourArea(current, CV_WHOLE_SEQ, 0);
        //printf("Area do contorno: %.2f\n", area);
        
        //Desenha os contornos da imagem quando a area for acima ou igual a 303212.00 (valor que peguei de uma area média das imagens) 
        if (area > 1000) {
            float perimetro = cvArcLength(current, CV_WHOLE_SEQ, 0);

            // Criar um novo armazenamento para a aproximação do polígono
            CvMemStorage* storage_poly = cvCreateMemStorage(0);

            // Criar uma sequência para armazenar os contornos aproximados
            CvSeq* approx = cvApproxPoly(
                current,             // Contorno de entrada
                sizeof(CvContour),   // Tamanho do contorno
                storage_poly,        // Armazenamento da memória
                CV_POLY_APPROX_DP,   // Tipo de aproximação (Douglas-Peucker)
                0.0005 * perimetro,   // Precisão da aproximação (2% do perímetro)
                1                    // Fechar o contorno (1 = fechado, 0 = aberto)
            );

            // Desenhar os contornos aproximados na imagem original
            cvDrawContours(img, approx, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), 0, 2, 8, cvPoint(0,0));

            // Liberar memória do armazenamento usado
            cvReleaseMemStorage(&storage_poly);
        }
        
    }

    return(img, num_contours);

    // Liberar memória
    cvReleaseMemStorage(&storage);
}

int main() {
    char imgPath[256]; 
    int contaImg = 1; // Contador inicial para as imagens
    const char* grayWindow = "Imagem em Escala de Cinza";
    const char* dilateWindow = "Imagem Dilatada";

    // Criar janelas para exibir as imagens
    //cvNamedWindow(grayWindow, CV_WINDOW_AUTOSIZE);
    //cvNamedWindow(dilateWindow, CV_WINDOW_AUTOSIZE);

    while (1) {
        // Gerar o caminho da imagem
        snprintf(imgPath, sizeof(imgPath), "C:\\Dev\\tcc_project_c\\image\\imagem%d.jpg", contaImg);

        // Carregar a imagem original
        IplImage* img = cvLoadImage(imgPath, CV_LOAD_IMAGE_COLOR);
        if (img == NULL) {
            //printf("Nao foi possivel carregar a imagem: %s\n", imgPath);
            break; // Encerrar o loop ao atingir o final da sequência
        }

        // Converter para escala de cinza
        IplImage* imgGray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
        cvCvtColor(img, imgGray, CV_BGR2GRAY);

        // Aplicar limiarização para binarizar a imagem
        IplImage* imgThreshold = cvCreateImage(cvGetSize(imgGray), IPL_DEPTH_8U, 1);
        cvAdaptiveThreshold(imgGray, imgThreshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, 1);

        // Aplicar dilatação para melhorar a detecção de contornos
        IplImage* imgDil = cvCreateImage(cvGetSize(imgThreshold), IPL_DEPTH_8U, 1);
        cvDilate(imgThreshold, imgDil, NULL, 1);

        getContours(imgDil, img);

        // Exibir imagens redimensionadas para verificação
        //showAndWaitResized(grayWindow, imgGray, imgGray->width / 2, imgGray->height / 2); //  Reduzido pela metade
        //showAndWaitResized(dilateWindow, imgDil, imgDil->width / 2, imgDil->height / 2); //  Reduzido pela metade
        showAndWaitResized(dilateWindow, img, img->width / 2, img->height / 2);

        // Processar os contornos
        

        // Liberar memória para as imagens processadas
        cvReleaseImage(&img);
        cvReleaseImage(&imgGray);
        cvReleaseImage(&imgThreshold);
        cvReleaseImage(&imgDil);

        // Incrementar o contador
        contaImg++;
    }

    // Limpar recursos
    cvDestroyAllWindows();

    return 0;
}
