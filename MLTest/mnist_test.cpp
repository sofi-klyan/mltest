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

int DataSet::CopyOutputs(unsigned char* pOutputs, int size)
{
	if (size != m_sampleNumber)
	{
		return e_Error;
	}

	if (m_pOutputs)
	{
		delete (m_pOutputs);
	}
	m_pOutputs = new unsigned char[size];
	memcpy(m_pOutputs, pOutputs, size * sizeof(unsigned char));

	return e_OK;
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

int MNISTTester::LoadDataSubSet(
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
	
	pSet->m_featureNumber = set.m_featureNumber;
	Preprocessor::ChooseDataSubset(
		set.m_ppInputs,
		set.m_pOutputs,
		set.m_sampleNumber,
		set.m_featureNumber,
		maxSampleNumber,
		&pSet->m_ppInputs,
		&pSet->m_pOutputs,
		&pSet->m_sampleNumber
	);

	return e_OK;

}

int MNISTTester::PerformPCA(
	const DataSet& subset,
	DataSet* pSet,
	char* pPCAOutputFile
)
{
	if (!pSet)
	{
		return e_Error;
	}

	DataSet* pReducedSet = pSet;
	pReducedSet->m_featureNumber = m_pcNum;
	pReducedSet->m_sampleNumber = subset.m_sampleNumber;
	pReducedSet->CopyOutputs(subset.m_pOutputs, pReducedSet->m_sampleNumber);
	
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
		pPCAOutputFile
	);

	return e_OK;
}

int MNISTTester::Process()
{
	DataSet trainSet, testSet;
	DataSet trainReducedSet, testReducedSet;

	char* pcaMatrixFile = "pca_mat.txt";
	char* kNNFile = "knn.txt";

	// preprocess
	LoadDataSubSet(m_pTrainInputFile, m_pTrainOutputFile, m_maxTrainSampleNum, &trainSet);
	PerformPCA(trainSet, &trainReducedSet, pcaMatrixFile);

	// train
	KNNClassifier* pKNNCl = new KNNClassifier(m_classNum);
	
	pKNNCl->Train(
		(void**)trainReducedSet.m_ppInputs,
		(void*)trainReducedSet.m_pOutputs,
		trainReducedSet.m_sampleNumber,
		m_pcNum
	);	
	
	/*
	pKNNCl->SetClassifier(
		(void**)trainReducedSet.m_ppInputs, 
		(void*)trainReducedSet.m_pOutputs, 
		trainReducedSet.m_sampleNumber, 
		m_pcNum,
		3
	);
	*/
	pKNNCl->Save(kNNFile);

	delete (pKNNCl);
	pKNNCl = new KNNClassifier();
	pKNNCl->Load(kNNFile);

	// test
	LoadDataSubSet(m_pTestInputFile, m_pTestOutputFile, m_maxTestSampleNum, &testSet);
	testReducedSet.m_sampleNumber = testSet.m_sampleNumber;
	testReducedSet.CopyOutputs(testSet.m_pOutputs, testSet.m_sampleNumber);

	Preprocessor::ReduceByPCAMatrix(
		(const unsigned char**)testSet.m_ppInputs,
		testSet.m_sampleNumber,
		testSet.m_featureNumber,
		pcaMatrixFile,
		&testReducedSet.m_ppInputs,
		&testReducedSet.m_featureNumber
	);

	int testError = 0;
	pKNNCl->Test((void**)testReducedSet.m_ppInputs, testReducedSet.m_pOutputs, testReducedSet.m_sampleNumber, &testError);
	string output = "Test Set error: " + to_string(testError) + " out of " + to_string(testReducedSet.m_sampleNumber);
	Logger::PrintLog(output.c_str());
	
	// clean
	delete (pKNNCl);

	return e_OK;

}