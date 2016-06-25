#pragma once
#include "opencv2/opencv.hpp"

class Preprocessor {

public:
	Preprocessor();
	~Preprocessor();

	static int ChooseDataSubset(
		unsigned char** ppInputs,
		unsigned char* pOutputs,
		int sampleNumber,
		int featureNum,
		const int maxSubsetSize,
		unsigned char*** pppSubsetInputs,
		unsigned char** ppSubsetOutputs,
		int* pSubsetSize
	);

	static int calcPCA(
		const unsigned char** ppInputSamples,
		int sampleNumber,
		int featureNumber,
		int pcNum,
		unsigned char**ppConvertedBackSamples,
		unsigned char**ppReducedSamples,
		char* pMatrixFile
	);

	static int ReduceByPCAMatrix(
		const unsigned char** ppInputSamples,
		int sampleNumber,
		int featureNumber, 
		char* pMatrixFile,
		unsigned char*** pppReducedSamples,
		int* pReducedFeatureNumber
	);

private:
	static int SaveMatrix(char* pMatrixFile, cv::Mat& m);
	static int LoadMatrix(char* pMatrixFile, cv::Mat& m);
};