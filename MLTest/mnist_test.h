#pragma once

class DataSet
{
public:
	DataSet();
	~DataSet();
	void Clean();
	int CopyOutputs(unsigned char* pOutputs, int size);

private:
	void CleanInputs();
public:
	unsigned char** m_ppInputs;
	unsigned char* m_pOutputs;
	int m_sampleNumber;
	int m_featureNumber;
};

class MNISTTester
{
public:
	MNISTTester(
		const char* pTrainInputFile,
		const char* pTrainOutputFile,
		const char* pTestInputFile,
		const char* pTestOutputFile,
		int maxTrainSampleNum,
		int maxTestSampleNum
		);
	~MNISTTester();

	int Process();


private:
	MNISTTester();

	int LoadDataSet(const char* pInputFile, const char* pOutputFile, DataSet* pSet);
	int LoadDataSubSet(
		const char* pInputFile,
		const char* pOutputFile,
		int maxSampleNumber,
		DataSet* pSet
	);
	int PerformPCA(
		const DataSet& subset,
		DataSet* pSet,
		char* pPCAOutputFile
	);

private:
	const int m_classNum = 10;
	const int m_pcNum = 70;
	int m_featureNum;
	char* m_pTrainInputFile;
	char* m_pTrainOutputFile;
	char* m_pTestInputFile;
	char* m_pTestOutputFile;
	int m_maxTrainSampleNum;
	int m_maxTestSampleNum;
};

