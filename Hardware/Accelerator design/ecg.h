#ifndef _ECG_H
#define _ECG_H

#include<iostream>
#include<sstream>
#include <string.h>
#include "ap_fixed.h"


using namespace std;


#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define S 1                       
#define K 5						

#define Tn 32		
#define Tm 32		
#define Tr 19		

#define Kernel_size 5

#define OnChipInBuf_Height ((Tr-1)*S+Kernel_size)	
#define MAX_BIAS_LENGTH (512)



typedef  unsigned char dtype;
typedef ap_fixed<32, 2, AP_RND, AP_WRAP >  m_dtype;


void ECG_Stage( dtype *input, dtype *output, dtype *weight, int *bias, dtype *zero_p, m_dtype *M, const int Kernel_stride, const int InFM_num, const int OutFM_num,
				const int input_row, const int output_h,
				int mLoops, const int nLoops, const int rLoops, int weight_row);


#endif
