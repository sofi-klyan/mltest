#include "mnist_test.h"
#include "io.h"
#include "preprocess.h"
#include "train.h"
#include "err.h"

using namespace std;

DataSet::DataSet()
{
	m_ppInputs = NULL;
	m_pOutputs = NULL;
	m_featureNumber = 0;
	m_sampleNumber = 0;
}

DataSet::~DataSet()
{
	Clean();
}

void DataSet::Clean()
{
	CleanInputs();
	if (m_pOutputs)
	{
		delete(m_pOutputs);
	}
	m_pOutputs = NULL;
	m_featureNumber = 0;
	m_sampleNumber = 0;
}

void DataSet::CleanInputs()
{
	if (m_ppInputs)
	{
		for (int i = 0; i < m_sampleNumber; i++)
		{
			if (m_ppInputs[i])
			{
				delete (m_ppInputs[i]);
			}
		}
		delete m_ppInputs;
		m_ppInputs = NULL;
	}
}


MNISTTester::MNISTTester()
{
}

MNISTTester::~MNISTTester()
{
}

MNISTTester::MNISTTester(
	const char* pTrainInputFile,
	const char* pTrainOutputFile,
	const char* pTestInputFile,
	const char* pTestOutputFile,
	int maxTrainSampleNum = 2000,
	int maxTestSampleNum = 2000
)
{
	m_pTestInputFile = (char*) pTestInputFile;
	m_pTestOutputFile = (char*)pTestOutputFile;
	m_pTrainInputFile = (char*)pTrainInputFile;
	m_pTrainOutputFile = (char*)pTrainOutputFile;

	m_maxTestSampleNum = maxTestSampleNum;
	m_maxTrainSampleNum = maxTrainSampleNum;
}

int MNISTTester::LoadDataSet(const char* pInputFile, const char* pOutputFile, DataSet* pSet)
{
	if (!pSet)
	{
		return e_Error;
	}

	int imgWidth = 0, imgHeight = 0;
	int outputSampleNumber;

	// read images	
	MNISTReader::ReadImages((char*)pInputFile, &pSet->m_ppInputs, &pSet->m_sampleNumber, &imgWidth, &imgHeight);

	// read labels	
	MNISTReader::ReadLabels((char*)pOutputFile, &pSet->m_pOutputs, &outputSampleNumber);

	if (outputSampleNumber != pSet->m_sampleNumber)
	{
		//cout << "img number mismatch" << endl;
		return e_Error;
	}

	pSet->m_featureNumber = imgWidth * imgHeight;

	return e_OK;
}

int MNISTTester::LoadAndPreprocessData(
	const char* pInputFile, 
	const char* pOutputFile, 
	int maxSampleNumber, 
	DataSet* pSet
)
{
	if (!pSet)
	{
		return e_Error;
	}

	DataSet set;
	if (e_OK != LoadDataSet(pInputFile, pOutputFile, &set))
	{
		return e_Error;
	}

	DataSet subset;
	subset.m_featureNumber = set.m_featureNumber;
	Preprocessor::ChooseDataSubset(
		set.m_ppInputs,
		set.m_pOutputs,
		set.m_sampleNumber,
		set.m_featureNumber,
		maxSampleNumber,
		&subset.m_ppInputs,
		&subset.m_pOutputs,
		&subset.m_sampleNumber
	);

	// calcPCA
	DataSet* pReducedSet = pSet;
	pReducedSet->m_featureNumber = m_pcNum;	
	pReducedSet->m_sampleNumber = subset.m_sampleNumber;	
	pReducedSet->m_pOutputs = new unsigned char[pReducedSet->m_sampleNumber];
	for (int i = 0; i < pReducedSet->m_sampleNumber; i++)
	{
		pReducedSet->m_pOutputs[i] = subset.m_pOutputs[i];
	}
	pReducedSet->m_ppInputs = new unsigned char*[subset.m_sampleNumber];	
	for (int i = 0; i < subset.m_sampleNumber; i++)
	{
		pReducedSet->m_ppInputs[i] = new unsigned char[m_pcNum];
	}
	Preprocessor::calcPCA(
		(const unsigned char**)subset.m_ppInputs,
		subset.m_sampleNumber,
		subset.m_featureNumber,
		m_pcNum,
		NULL,
		pReducedSet->m_ppInputs,
		"m.txt"
	);

	// clean
	/*
	if (set.m_ppInputs)
	{
		delete (set.m_ppInputs);
	}
	if (set.m_pOutputs)
	{
		delete (set.m_pOutputs);
	}
	if (subset.m_ppInputs)
	{
		delete (subset.m_ppInputs);
	}
	*/

	return e_OK;
}

int MNISTTester::Process()
{
	DataSet trainSet;

	// preprocess
	LoadAndPreprocessData(m_pTrainInputFile, m_pTrainOutputFile, m_maxTrainSampleNum, &trainSet);

	// train
	KNNClassifier* pKNNCl = new KNNClassifier(m_classNum);
	pKNNCl->Train((void**)trainSet.m_ppInputs, (void*)trainSet.m_pOutputs, trainSet.m_sampleNumber, m_pcNum);
	pKNNCl->Save("res_k.txt"); // not fully implemented yet

	// clean
	delete (pKNNCl);

	return e_OK;

}