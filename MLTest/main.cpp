#include "mnist_test.h"

int main(int argc, char* argv[])
{
	char* imagesFile = "D:\\IT\\Data\\train-images.idx3-ubyte";
	char* testImagesFile = "D:\\IT\\Data\\t10k-images.idx3-ubyte";
	char* labelsFile = "D:\\IT\\Data\\train-labels.idx1-ubyte";
	char* testLabelsFile = "D:\\IT\\Data\\t10k-labels.idx1-ubyte";
	
	const int max_img_number = 2000;

	MNISTTester* test = new MNISTTester(imagesFile, labelsFile, testImagesFile, testLabelsFile, max_img_number, max_img_number);
	test->Process();
	
	return 0;
}