#include "preprocess.h"
#include "err.h"
#include "io.h"
#include "time.h"
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

int Preprocessor::ChooseDataSubset(
	unsigned char** ppInputs,
	unsigned char* pOutputs,
	int sampleNumber,
	int featureNum,
	const int maxSubsetSize,
	unsigned char*** pppSubsetInputs,
	unsigned char** ppSubsetOutputs,
	int* pSubsetSize
)
{
	if (!ppInputs || !pOutputs ||
		!pppSubsetInputs || !ppSubsetOutputs ||
		!pSubsetSize ||
		(sampleNumber <= 0))
	{
		return e_Error;
	}

	// choose subset
	bool isSetTooBig = (maxSubsetSize > 0) && (sampleNumber > maxSubsetSize);
	int subsetSampleNumber = isSetTooBig ? maxSubsetSize : sampleNumber;
	unsigned char** ppSubsetInputs = new unsigned char*[subsetSampleNumber];
	unsigned char* pSubsetOutputs = new unsigned char[subsetSampleNumber];
	int* randIndArray = NULL;

	if (isSetTooBig)
	{
		int item;
		randIndArray = new int[maxSubsetSize];
		for (int i = 0; i < maxSubsetSize; i++)
		{
			bool unique;
			do
			{
				unique = true;
				item = rand() % sampleNumber + 1;
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

	for (int i = 0; i < subsetSampleNumber; i++)
	{
		int ind;

		if (isSetTooBig)
		{
			srand(time(NULL));
			ind = randIndArray[i];//rand() % sampleNumber;

		}
		else
		{
			ind = i;
		}
		ppSubsetInputs[i] = new unsigned char[featureNum];
		memcpy(ppSubsetInputs[i], ppInputs[ind], sizeof(unsigned char) * featureNum);
		pSubsetOutputs[i] = pOutputs[ind];
	}

	if (randIndArray)
	{
		delete (randIndArray);
	}

	*pppSubsetInputs = ppSubsetInputs;
	*ppSubsetOutputs = pSubsetOutputs;
	*pSubsetSize = subsetSampleNumber;

	return e_OK;
};

int Preprocessor::calcPCA(
	const unsigned char** ppInputSamples, 
	int sampleNumber, 
	int featureNumber, 
	int pcNum, 
	unsigned char** ppConvertedBackSamples, 
	unsigned char** ppReducedSamples,
	char* pMatrixFile
)
{
	if (!ppInputSamples || (sampleNumber <= 0) || (pcNum <= 0))
	{
		return e_Error;
	}

	Mat data_pts = Mat(sampleNumber, featureNumber, CV_64FC1);
	for (int i = 0; i < sampleNumber; i++)
	{
		for (int j = 0; j < featureNumber; j++)
		{
			data_pts.at<double>(i, j) = ppInputSamples[i][j];
		}
	}


	Mat cov, mu;
	calcCovarMatrix(data_pts, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	cov = cov / (data_pts.rows - 1);

	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);

	Mat eigenMat = Mat(featureNumber, pcNum, CV_64FC1);
	for (int i = 0; i < pcNum; i++)
	{
		for (int j = 0; j < featureNumber; j++)
		{
			eigenMat.at<double>(j, i) = pca_analysis.eigenvectors.at<double>(i, j);
		}
	}

	Mat shifted_input = Mat(sampleNumber, featureNumber, CV_64FC1);
	for (int i = 0; i < sampleNumber; i++)
	{
		for (int j = 0; j < featureNumber; j++)
		{
			shifted_input.at<double>(i, j) = data_pts.at<double>(i, j) - pca_analysis.mean.at<double>(0, j);
		}
	}

	Mat reduced_input = Mat(sampleNumber, pcNum, CV_64FC1);
	reduced_input = shifted_input * eigenMat;

	if (ppConvertedBackSamples)
	{
		Mat recovered_input = Mat(sampleNumber, pcNum, CV_64FC1);
		recovered_input = reduced_input * eigenMat.t();

		for (int i = 0; i < sampleNumber; i++)
		{
			for (int j = 0; j < featureNumber; j++)
			{
				int val = (int)round(recovered_input.at<double>(i, j) + pca_analysis.mean.at<double>(0, j));
				if (val < 0) val = 0;
				else if (val > 255) val = 255;
				ppConvertedBackSamples[i][j] = (unsigned char)val;
			}

		}
	}

	if (ppReducedSamples)
	{
		for (int i = 0; i < sampleNumber; i++)
		{
			for (int j = 0; j < pcNum; j++)
			{
				int val = (int)round(reduced_input.at<double>(i, j));
				if (val < 0) val = 0;
				else if (val > 255) val = 255;
				ppReducedSamples[i][j] = (unsigned char)val;
			}

		}
	}

	SaveMatrix(pMatrixFile, eigenMat);
	
	return e_OK;
}

int Preprocessor::ReduceByPCAMatrix(
	const unsigned char** ppInputSamples,
	int sampleNumber,
	int featureNumber,
	char* pMatrixFile,
	unsigned char*** pppReducedSamples,
	int* pReducedFeatureNumber
)
{
	if (!ppInputSamples || !pppReducedSamples || !pReducedFeatureNumber)
	{
		return e_Error;
	}

	Mat eigenMat;
	if (e_OK != LoadMatrix(pMatrixFile, eigenMat))
	{
		return e_Error;
	}
	int reducedFeatureNumber = eigenMat.cols;

	Mat data_pts = Mat(sampleNumber, featureNumber, CV_64FC1);
	for (int i = 0; i < sampleNumber; i++)
	{
		for (int j = 0; j < featureNumber; j++)
		{
			data_pts.at<double>(i, j) = ppInputSamples[i][j];
		}
	}

	Mat mu = Mat(data_pts.rows, data_pts.cols, CV_64FC1);
	for (int i = 0; i < mu.cols; i++)
	{
		double meanColValue = mean(data_pts.col(i)).val[0];
		for (int j = 0; j < mu.rows; j++)
		{
			mu.at<double>(j, i) = meanColValue;
		}
	}

	data_pts = data_pts - mu;

	Mat reduced_pts = Mat(sampleNumber, reducedFeatureNumber, CV_64FC1);
	reduced_pts = data_pts * eigenMat;

	unsigned char** ppReducedSamples = new unsigned char* [sampleNumber];
	for (int i = 0; i < sampleNumber; i++)
	{
		ppReducedSamples[i] = new unsigned char[reducedFeatureNumber];
		for (int j = 0; j < reducedFeatureNumber; j++)
		{
			int val = (int)round(reduced_pts.at<double>(i, j));
			if (val < 0) val = 0;
			else if (val > 255) val = 255;
			ppReducedSamples[i][j] = (unsigned char)val;
		}

	}

	*pppReducedSamples = ppReducedSamples;
	*pReducedFeatureNumber = reducedFeatureNumber;

	return e_OK;
}

int Preprocessor::LoadMatrix(char* pMatrixFile, cv::Mat& m)
{
	if (pMatrixFile)
	{
		ifstream fin(pMatrixFile);

		if (!fin)
		{
			string log = string("File can not be opened for loading a matrix: ") + pMatrixFile;
			Logger::PrintLog(log.c_str());
			return e_Error;
		}

		int rows, cols;
		fin >> rows;
		fin >> cols;
		m = Mat(rows, cols, CV_64FC1);
		for (int i = 0; i < m.rows; i++)
		{
			for (int j = 0; j < m.cols; j++)
			{
				fin >> m.at<double>(i, j);
			}			
		}

		fin.close();
	}
	else
	{
		return e_Error;
	}

	return e_OK;
}

/* Saving Mat with double values in the following format:
** <rows number> <cols number>
** m(0,0) m(0, 1) ... m(0, cols)
** ....
** m(rows, 0) ...     m(rows, cols)
*/
int Preprocessor::SaveMatrix(char* pMatrixFile, Mat& m)
{
	if (pMatrixFile)
	{
		ofstream fout(pMatrixFile);

		if (!fout)
		{			
			string log = string("File can not be opened for saving a matrix: ") + pMatrixFile;
			Logger::PrintLog(log.c_str());
			return e_Error;
		}

		fout << m.rows << "\t";
		fout << m.cols << endl;
		for (int i = 0; i < m.rows; i++)
		{
			for (int j = 0; j < m.cols; j++)
			{
				fout << m.at<double>(i, j) << "\t";
			}
			fout << endl;
		}

		fout.close();
	}
	else
	{
		return e_Error;
	}

	return e_OK;
}

Preprocessor::Preprocessor()
{}

Preprocessor::~Preprocessor()
{}