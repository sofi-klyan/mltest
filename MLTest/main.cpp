#include "mnist_test.h"
#include <string>

using namespace std;

/* usage ex: <app_name> --train-inputs D:\IT\Data\train-images.idx3-ubyte --train-outputs D:\IT\Data\train-labels.idx1-ubyte  
--test-inputs D:\IT\Data\t10k-images.idx3-ubyte --test-outputs D:\IT\Data\t10k-labels.idx1-ubyte 
*/

int main(int argc, char* argv[])
{
	char* imagesFile = "D:\\IT\\Data\\train-images.idx3-ubyte";
	char* testImagesFile = "D:\\IT\\Data\\t10k-images.idx3-ubyte";
	char* labelsFile = "D:\\IT\\Data\\train-labels.idx1-ubyte";
	char* testLabelsFile = "D:\\IT\\Data\\t10k-labels.idx1-ubyte";
	
	int max_img_number = 2000;

	for (int i_argNum = 1; i_argNum < argc; ++i_argNum)
	{
		if (string(argv[i_argNum]) == "--train-inputs")
		{
			imagesFile = argv[++i_argNum];
		}
		else if (string(argv[i_argNum]) == "--train-outputs")
		{
			labelsFile = argv[++i_argNum];
		}
		if (string(argv[i_argNum]) == "--test-inputs")
		{
			testImagesFile = argv[++i_argNum];
		}
		else if (string(argv[i_argNum]) == "--test-outputs")
		{
			testLabelsFile = argv[++i_argNum];
		}
	}
	
	MNISTTester* test = new MNISTTester(imagesFile, labelsFile, testImagesFile, testLabelsFile, max_img_number, max_img_number);
	test->Process();
	
	return 0;
}