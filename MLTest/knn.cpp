#include "train.h"
#include "err.h"
#include "io.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

KNNClassifier::KNNClassifier()
{
	m_k = 0;

	m_ppTrainInputs = NULL;
	m_pTrainOutputs = NULL;
}

KNNClassifier::KNNClassifier(int _classNum)
{
	m_k = 0;
	m_classNum = _classNum;

	m_ppTrainInputs = NULL;
	m_pTrainOutputs = NULL;
}

KNNClassifier::~KNNClassifier ()
{};

int KNNClassifier::Save(char* fileName)
{
	eErrorCode res = e_OK;

	ofstream resFile;
	resFile.open(fileName);
	resFile << m_k;
	resFile.close();

	return res;
}

int KNNClassifier::Load(char* fileName)
{
	eErrorCode res = e_OK;

	//TODO - add format check
	fstream file(fileName, std::ios_base::in);
	file >> m_k;
	
	return res;
}

int KNNClassifier::Train(void** pTrainInputs, void* pTrainOutputs, int sampleNumber, int featureNumber)
{
	if (!pTrainInputs || !pTrainOutputs || (sampleNumber <= 0))
	{
		return e_Error;
	}

	unsigned char** ppImages = (unsigned char**)pTrainInputs;
	unsigned char* pLabels = (unsigned char*)pTrainOutputs;

	SetTrainSet((unsigned char**)pTrainInputs, (unsigned char*)pTrainOutputs, sampleNumber, featureNumber);

	// calc metrics
	vector<metricElem>* pMetricVec = new vector<metricElem>[m_trainSampleNum];
	calcTrainMetrics(pMetricVec);

	// get optimal k
	m_k = getOptimalK(pMetricVec);

	return e_OK;
}

int KNNClassifier::Test(void** ppTestInputs, void* pTestOutputs, int sampleNumber, int* pError)
{
	if (!m_ppTrainInputs || !m_pTrainOutputs || 
		!ppTestInputs || !pTestOutputs ||
		!pError || (sampleNumber <= 0) )
	{
		return e_Error;
	}

	int error = 0;
	unsigned char* pOutputs = (unsigned char*)pTestOutputs;

	for (int i = 0; i < sampleNumber; i++)
	{
		unsigned char output;
		if (e_OK != Classify(ppTestInputs[i], &output))
		{
			return e_Error;
		}

		if (output != pOutputs[i])
		{
			error++;
		}
	}

	*pError = error;

	return e_OK;
}

int KNNClassifier::Classify(void* pInput, void* pOutput)
{
	if (!pInput || !pOutput || m_ppTrainInputs || m_pTrainOutputs)
	{
		return e_Error;
	}

	unsigned char* pInputSample = (unsigned char*)pInput;
	unsigned char outputClass = 0;

	vector<metricElem> metricVec;

	for (int j = 0; j < m_trainSampleNum; j++)
	{		
		long metric = calcMetric(pInputSample, m_ppTrainInputs[j], m_featureNum);
		metricElem mtr(metric, j);
		metricVec.push_back(mtr);
	}
	sort(metricVec.begin(), metricVec.end(), less_than_key());

	outputClass = (unsigned char) getKNNClass(&metricVec, m_pTrainOutputs, m_k, m_classNum);
	*((unsigned char*)pOutput) = outputClass;

	return e_OK;
}

int KNNClassifier::SetTrainSet(unsigned char** pTrainInputs, unsigned char* pTrainOutputs, int sampleNumber, int featureNumber)
{
	if (!pTrainInputs || !pTrainOutputs || (sampleNumber <= 0))
	{
		return e_Error;
	}
	
	if (m_ppTrainInputs)
	{
		delete (m_ppTrainInputs);
	}
	if (m_pTrainOutputs)
	{
		delete (m_pTrainOutputs);
	}
	m_ppTrainInputs = new unsigned char*[sampleNumber];
	m_pTrainOutputs = new unsigned char[sampleNumber];
	for (int i = 0; i < sampleNumber; i++)
	{
		m_ppTrainInputs[i] = new unsigned char[featureNumber];
		for (int j = 0; j < featureNumber; j++)
		{
			m_ppTrainInputs[i][j] = pTrainInputs[i][j];
		}
		m_pTrainOutputs[i] = pTrainOutputs[i];
	}

	m_trainSampleNum = sampleNumber;
	m_featureNum = featureNumber;

	return e_OK;
}

long KNNClassifier::calcMetric(unsigned char* pImg1, unsigned char* pImg2, int size)
{
	long metric = 0;
	for (int i = 0; i < size; i++)
	{
		int pixSqrDiff = (pImg1[i] - pImg2[i]) * (pImg1[i] - pImg2[i]);
		metric += pixSqrDiff;
	}

	return metric;
}

void KNNClassifier:: calcTrainMetrics(	
	vector<metricElem>* pMetricVec
)
{
	for (int i = 0; i < m_trainSampleNum; i++)
	{
		for (int j = 0; j < m_trainSampleNum; j++)
		{
			if (i == j) continue;
			long metric = calcMetric(m_ppTrainInputs[i], m_ppTrainInputs[j], m_featureNum);
			metricElem mtr(metric, j);
			pMetricVec[i].push_back(mtr);
		}
		sort(pMetricVec[i].begin(), pMetricVec[i].end(), less_than_key());
	}
}

int KNNClassifier::getKNNClass(vector<metricElem>* pMetricVec, unsigned char* pLabels, int k, int class_number)
{
	vector<int> belongToClassArr(class_number);

	for (int i_neighbInd = 0; i_neighbInd < k; i_neighbInd++)
	{
		int imgInd = pMetricVec->at(i_neighbInd).imgInd;
		belongToClassArr[pLabels[imgInd]]++;
	}
	auto biggest = max_element(begin(belongToClassArr), end(belongToClassArr));
	return  distance(begin(belongToClassArr), biggest);
}

int KNNClassifier::getSetError(vector<metricElem>* pMetricVec, unsigned char* pLabels, int k, int img_number, int class_number)
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


int KNNClassifier::getOptimalK(vector<metricElem>* pMetricVec)
{
	int k_step = 1;// img_number / 10;
	int k;
	int min_error = INT_MAX;
	int min_error_k = -1;

	for (k = 1; k < m_trainSampleNum; k += k_step)
	{
		int k_error = getSetError(pMetricVec, m_pTrainOutputs, k, m_trainSampleNum, m_classNum);

		if (k_error < min_error)
		{
			min_error = k_error;
			min_error_k = k;
		}
	}

	// log
	string log = string ("optimal k : ") + to_string(min_error_k) + string(" error = ") + to_string(min_error);
	Logger::PrintLog(log.c_str());
	
	return min_error_k;
}


