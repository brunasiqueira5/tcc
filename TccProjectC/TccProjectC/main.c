#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <stdio.h>

int main() {
    char imgPath[256];
    int contaImg = 1; // Contador inicial para as imagens
    const char* windowName = "Imagem em Escala de Cinza";

    // Criar janela para exibir as imagens
    cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);

    while (1) {
        // Gerar o caminho da imagem
        snprintf(imgPath, sizeof(imgPath), "C:\\Dev\\tcc\\images\\imagem%d.jpg", contaImg);

        // Carregar a imagem em escala de cinza
        IplImage* img = cvLoadImage(imgPath, CV_LOAD_IMAGE_GRAYSCALE);

        // Verificar se a imagem foi carregada com sucesso
        if (img == NULL) {
            printf("Não foi possível carregar a imagem: %s\n", imgPath);
            break; // Encerrar o loop ao atingir o final da sequência
        }

        // Exibir a imagem
        cvShowImage(windowName, img);

        // Aguardar uma tecla para exibir a próxima imagem
        int key = cvWaitKey(0);
        if (key == 27) { // Tecla ESC para sair
            break;
        }

        // Liberar a imagem atual antes de carregar a próxima
        cvReleaseImage(&img);

        // Incrementar o contador
        contaImg++;
    }

    // Limpar recursos
    cvDestroyAllWindows();

    return 0;
}
