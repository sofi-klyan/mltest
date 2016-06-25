#pragma once


class Logger
{
public:
	static int PrintLog(const char* msg);
};

class MNISTReader
{
public:
	MNISTReader();
	~MNISTReader();

	static void ReadImages(
		char* pFileName,
		unsigned char*** pppArr,
		int* pImgNumber,
		int* pImgWidth,
		int* pImgHeight
	);

	static void ReadLabels(
		char* pFileName,
		unsigned char** ppArr,
		int* pImgNumber
	);

private:
	static int ReverseInt(int i);

};

