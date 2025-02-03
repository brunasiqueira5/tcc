#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <opencv2/core/core_c.h> 

	void showAndWaitResized(const char* windowName, IplImage* img, int width, int height);
	void* createSVM();
	void trainSVM(void* svm);
	float predictSVM(void* svm, IplImage* img);
	void detectTardigrades(void* svm, IplImage* imgDil, IplImage* img); 

#ifdef __cplusplus
}
#endif