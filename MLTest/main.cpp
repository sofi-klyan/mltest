#include "stdio.h"
#include "string.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <algorithm>
#include <stdlib.h>
#include "time.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

typedef struct _metricElem
{
	long metric;
	int imgInd;

	_metricElem(long _metric, int _imgInd) : metric(_metric), imgInd(_imgInd) {}

	bool operator > (const _metricElem & mtr) const
	{
		return (metric > mtr.metric);
	}
} metricElem;

struct less_than_key
{
	inline bool operator() (const metricElem& struct1, const metricElem& struct2)
	{
		return (struct1.metric < struct2.metric);
	}
};

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMNISTImages(
	char* pFileName, 
	unsigned char*** pppArr,
	int* pImgNumber,
	int* pImgWidth, 
	int* pImgHeight
	)
{	
	ifstream file(pFileName, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		int n_img_size = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		n_img_size = n_cols * n_rows;
		*pppArr = new unsigned char*[number_of_images];
		for (int i = 0; i<number_of_images; ++i)
		{
			(*pppArr)[i] = new unsigned char[n_img_size];
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					(*pppArr)[i][(n_rows*r) + c] = temp;
				}
			}
		}

		*pImgNumber = number_of_images;
		*pImgWidth = n_cols;
		*pImgHeight = n_rows;
	}
}

void ReadMNISTLabels(
	char* pFileName,
	unsigned char** ppArr,
	int* pImgNumber
	)
{
	ifstream file(pFileName, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		
		*ppArr = new unsigned char[number_of_images];
		for (int i = 0; i<number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			(*ppArr)[i] = temp;		
		}

		*pImgNumber = number_of_images;		
	}
}

long calcMetric(unsigned char* pImg1, unsigned char* pImg2, int width, int height)
{
	long metric = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int pixSqrDiff = (pImg1[i * width + j] - pImg2[i * width + j]) * (pImg1[i * width + j] - pImg2[i * width + j]);
			metric += pixSqrDiff;
		}
	}

	return metric;
}

long calcMetric(unsigned char* pImg1, unsigned char* pImg2, int size)
{
	long metric = 0;
	for (int i = 0; i < size; i++)
	{
		int pixSqrDiff = (pImg1[i] - pImg2[i]) * (pImg1[i] - pImg2[i]);
		metric += pixSqrDiff;
	}

	return metric;
}

void calcMetrics(unsigned char** pImgs, int imgNumber, int width, int height, vector<metricElem>* pMetricVec)
{	
	for (int i = 0; i < imgNumber; i++)
	{		
		for (int j = 0; j < imgNumber; j++)
		{
			if (i == j) continue;
			long metric = calcMetric(pImgs[i], pImgs[j], width, height);
			metricElem mtr (metric, j);		
			pMetricVec[i].push_back(mtr);
		}
		sort(pMetricVec[i].begin(), pMetricVec[i].end(), less_than_key());
	}
}

void calcMetrics(unsigned char** pImgs, int imgNumber, int featureNum, vector<metricElem>* pMetricVec)
{
	for (int i = 0; i < imgNumber; i++)
	{
		for (int j = 0; j < imgNumber; j++)
		{
			if (i == j) continue;
			long metric = calcMetric(pImgs[i], pImgs[j], featureNum);
			metricElem mtr(metric, j);
			pMetricVec[i].push_back(mtr);
		}
		sort(pMetricVec[i].begin(), pMetricVec[i].end(), less_than_key());
	}
}

void calcPCA(const unsigned char** pImgs, int imgNumber, int width, int height, int pcNum, unsigned char**ppOutputImgs, unsigned char**ppReducedImgs)
{
	int imgSize = width * height;
	Mat data_pts = Mat(imgNumber, imgSize, CV_64FC1);
	for (int i = 0; i < imgNumber; i++)
	{
		for (int j = 0; j < imgSize; j++)
		{
			data_pts.at<double>(i, j) = pImgs[i][j];
		}
	}


	Mat cov, mu;
	calcCovarMatrix(data_pts, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	cov = cov / (data_pts.rows - 1);

	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);

	Mat eigenMat = Mat(imgSize, pcNum, CV_64FC1);
	for (int i = 0; i < pcNum; i++)
	{
		for (int j = 0; j < imgSize; j++)
		{
			eigenMat.at<double>(j, i) = pca_analysis.eigenvectors.at<double>(i, j);
		}
	}

	Mat shifted_input = Mat(imgNumber, imgSize, CV_64FC1);
	for (int i = 0; i < imgNumber; i++)
	{
		for (int j = 0; j < imgSize; j++)
		{
			shifted_input.at<double>(i, j) = data_pts.at<double>(i, j) - pca_analysis.mean.at<double>(0, j);
		}
	}

	Mat reduced_input = Mat(imgNumber, pcNum, CV_64FC1);
	reduced_input = shifted_input * eigenMat;	
	
	if (ppOutputImgs)
	{
		Mat recovered_input = Mat(imgNumber, pcNum, CV_64FC1);
		recovered_input = reduced_input * eigenMat.t();

		for (int i = 0; i < imgNumber; i++)
		{
			for (int j = 0; j < imgSize; j++)
			{
				int val = (int)round(recovered_input.at<double>(i, j) + pca_analysis.mean.at<double>(0, j));
				if (val < 0) val = 0;
				else if (val > 255) val = 255;
				ppOutputImgs[i][j] = (unsigned char)val;
			}

		}
	}

	if (ppReducedImgs)
	{
		for (int i = 0; i < imgNumber; i++)
		{
			for (int j = 0; j < pcNum; j++)
			{
				int val = (int)round(reduced_input.at<double>(i, j));
				if (val < 0) val = 0;
				else if (val > 255) val = 255;
				ppReducedImgs[i][j] = (unsigned char)val;
			}

		}
	}
	

}

int getKNNClass(vector<metricElem>* pMetricVec, unsigned char* pLabels, int k, int class_number)
{
	vector<int> belongToClassArr (class_number);

	for (int i_neighbInd = 0; i_neighbInd < k; i_neighbInd++)
	{
		int imgInd = pMetricVec->at(i_neighbInd).imgInd;
		belongToClassArr[pLabels[imgInd]]++;
	}	
	auto biggest = max_element(begin(belongToClassArr), end(belongToClassArr));
	return  distance(begin(belongToClassArr), biggest);
}

int getSetError(vector<metricElem>* pMetricVec, unsigned char* pLabels, int k, int img_number, int class_number)
{
	int k_error = 0;
	for (int i_imgInd = 0; i_imgInd < img_number; i_imgInd++)
	{
		int y = getKNNClass(&pMetricVec[i_imgInd], pLabels, k, class_number);
		if (y != (int)pLabels[i_imgInd])
		{
			k_error++;
		}
	}

	return k_error;
}

int getOptimalK(vector<metricElem>* pMetricVec, unsigned char* pLabels, int img_number, int class_number)
{
	int k_step = 1;// img_number / 10;
	int k;
	int min_error = INT_MAX;
	int min_error_k = -1;

	for (k = 1; k < img_number; k += k_step)
	{
		int k_error = getSetError(pMetricVec, pLabels, k, img_number, class_number);

		if (k_error < min_error)
		{
			min_error = k_error;
			min_error_k = k;
		}
	}
	

	// log
	cout << "optimal k : " << min_error_k << " error = " << min_error;
	return min_error_k;
}



int main(int argc, char* argv[])
{
	char* imagesFile = "K:\\MNIST\\train-images.idx3-ubyte";
	int inputImgNumber = 0;
	int imgWidth = 0;
	int imgHeight = 0;
	int imgSize = 0;
	unsigned char** ppInputImgs = NULL;

	unsigned char** ppPcaImages = NULL;
	unsigned char** ppReducedImages = NULL;

	char* labelsFile = "K:\\MNIST\\train-labels.idx1-ubyte";
	int labelinputImgNumber = 0;
	unsigned char* pInputLabels = NULL;

	unsigned char** ppImgs = NULL;
	unsigned char* pLabels = NULL;
	int imgNumber = 0;

	const int class_number = 10;
	const int max_img_number = 2000;

	// read images	
	ReadMNISTImages(imagesFile, &ppInputImgs, &inputImgNumber, &imgWidth, &imgHeight);

	// read labels	
	ReadMNISTLabels(labelsFile, &pInputLabels, &labelinputImgNumber);

	if (labelinputImgNumber != inputImgNumber)
	{
		cout << "img number mismatch" << endl;
		return -1;
	}

	// choose subset
	bool isSetTooBig = (inputImgNumber > max_img_number);
	imgNumber =  isSetTooBig ? max_img_number : inputImgNumber;
	ppImgs = new unsigned char*[imgNumber];
	pLabels = new unsigned char[imgNumber];
	int* randIndArray = NULL;

	if (isSetTooBig)
	{
		int item;
		randIndArray = new int[max_img_number];
		for (int i = 0; i < max_img_number; i++)
		{
			bool unique;
			do
			{
				unique = true;
				item = rand() % inputImgNumber + 1;
				for (int i1 = 0; i1 < i; i1++)
				{
					if (randIndArray[i1] == item)
					{
						unique = false;
						break;
					}
				}
			} while (!unique);
			randIndArray[i] = item;
		}
	}	

	for (int i = 0; i <imgNumber; i++)
	{
		int ind;

		if (isSetTooBig)
		{
			srand(time(NULL));
			ind = randIndArray[i];//rand() % inputImgNumber;

		}
		else
		{
			ind = i;
		}		
		ppImgs[i] = new unsigned char[imgWidth * imgHeight];
		memcpy(ppImgs[i], ppInputImgs[ind], sizeof(unsigned char) * imgWidth * imgHeight);
		pLabels[i] = pInputLabels[ind];
	}	

	if (randIndArray)
	{
		delete (randIndArray);
	}

	// calcPCA	
	
	ppReducedImages = new unsigned char*[imgNumber];
	/*
	ppPcaImages = new unsigned char*[imgNumber];
	int pcaStep = 30;
	int minPCA = 70;//10;
	int maxPCA = 71;// imgNumber;
	
	for (int i = 0; i < imgNumber; i++)
	{
		ppPcaImages[i] = new unsigned char[imgWidth * imgHeight];
	}
	for (int pcNum = minPCA; pcNum < maxPCA; pcNum += pcaStep)
	{	
		calcPCA((const unsigned char**)ppImgs, imgNumber, imgWidth, imgHeight, pcNum, ppPcaImages, NULL);

		// save as img
		
		for (int i = 0; i < imgNumber; i++)
		{
			Mat m = Mat(imgHeight, imgWidth, CV_8UC1, ppPcaImages[i]);
			imwrite(to_string(i) + "_" + to_string(pcNum) + ".bmp", m);
			//m.release();
		}
		
		
	}
	for (int i = 0; i < imgNumber; i++)
	{
		delete (ppPcaImages[i]);
	}
	delete (ppPcaImages);
	*/

	int pcNum = 70;
	for (int i = 0; i < imgNumber; i++)
	{
		ppReducedImages[i] = new unsigned char[pcNum];
	}
	calcPCA((const unsigned char**)ppImgs, imgNumber, imgWidth, imgHeight, pcNum,  NULL, ppReducedImages);
	
	
	// calc metrics
	vector<metricElem>* pMetricVec = new vector<metricElem>[imgNumber];
	calcMetrics(ppReducedImages, imgNumber, pcNum, pMetricVec);

	// get optimal k
	int k_opt = getOptimalK(pMetricVec, pLabels, imgNumber, class_number);	

	// clean
	for (int i = 0; i < inputImgNumber; i++)
	{
		delete(ppInputImgs[i]);
	}
	delete(ppInputImgs);
	delete (pInputLabels);

	for (int i = 0; i < imgNumber; i++)
	{
		delete(ppImgs[i]);
	}
	delete(ppImgs);

	for (int i = 0; i < imgNumber; i++)
	{
		delete (ppReducedImages[i]);
	}
	delete (ppReducedImages);

	delete (pLabels);
	//delete (pMetricVec);

	return 0;
}