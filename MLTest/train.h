#pragma once
#include <vector>

using namespace std;

struct metricElem
{
	long metric;
	int imgInd;

	metricElem(long _metric, int _imgInd) : metric(_metric), imgInd(_imgInd) {}

	bool operator > (const metricElem & mtr) const
	{
		return (metric > mtr.metric);
	}
};

struct less_than_key
{
	inline bool operator() (const metricElem& struct1, const metricElem& struct2)
	{
		return (struct1.metric < struct2.metric);
	}
};

class Classifier
{
public:
	virtual int Train(void** ppTrainInputs, void* pTrainOutputs, int sampleNumber, int featureNumber) = 0;
	virtual int Test(void** ppTestInputs, void* pTestOutputs, int sampleNumber, int* pError) = 0;
	virtual int Classify(void* pInput, void* pOutput) = 0;
	virtual int Save(char* fileName) = 0;
	virtual int Load(char* fileName) = 0;	
};

class KNNClassifier : public Classifier
{
public:
	KNNClassifier();
	KNNClassifier(int _classNum);
	~KNNClassifier();

	int Train(void** ppTrainInputs, void* pTrainOutputs, int sampleNumber, int featureNumber);
	int Test(void** ppTestInputs, void* pTestOutputs, int sampleNumber, int* pError);
	int Classify(void* pInput, void* pOutput);
	int Save(char* fileName);
	int Load(char* fileName);	

private:

	int SetTrainSet(unsigned char** pTrainInputs, unsigned char* pTrainOutputs, int sampleNumber, int featureNumber);
	long calcMetric(unsigned char* pImg1, unsigned char* pImg2, int size);
	//void calcMetrics(unsigned char** pImgs, int imgNumber, int featureNum, vector<metricElem>* pMetricVec);
	void calcTrainMetrics(vector<metricElem>* pMetricVec);
	int getKNNClass(vector<metricElem>* pMetricVec, unsigned char* pLabels, int k, int class_number);
	int getSetError(vector<metricElem>* pMetricVec, unsigned char* pLabels, int k, int img_number, int class_number);
	int getOptimalK(vector<metricElem>* pMetricVec);

private:
	int m_k;
	int m_classNum;
	int m_trainSampleNum;
	int m_featureNum;
	unsigned char** m_ppTrainInputs;
	unsigned char* m_pTrainOutputs;
	
};

