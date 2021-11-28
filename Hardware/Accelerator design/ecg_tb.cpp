#include "ecg.h"
#include "stdlib.h"
#include "sample.h"
#include "pre_conv1.h"

using namespace std;

int main()

{
	int p,q;
	float sample_trans[12][5000]={0};

	for(p=0;p<12;p++){
		for(q=0;q<5000;q++){
			sample_trans[p][q]=sample[q][p];
		}
	}

	float sample_in[12][5019]={0};
	for(p=0;p<12;p++){
			for(q=0;q<5000;q++){
				sample_in[p][q+10]=sample_trans[p][q];
			}
		}



  	int i,j, retval=0;
  	dtype Output[32][2500] ={0};
  	dtype *input = &sample_in[0][0];
  	dtype *output =&Output[0][0] ;
  	dtype *weight =&pre_conv1_weight[0][0][0];
  	float *bias = &pre_conv1_bias[0];

  	int Kernel_stride=2;
  	int InFM_num=12;
  	int OutFM_num=32;
  	int input_row=5019;
	int weight_row = 252;
  	int output_h=2500;
  	int mLoops=1;
  	int nLoops=1;
  	int rLoops=25;

	ECG_FPGA(input, output, weight, bias, Kernel_stride, InFM_num, OutFM_num, input_row, output_h, mLoops, nLoops, rLoops, weight_row);
	
  	FILE *fp1;
    fp1=fopen("result.dat","w");
    for (i = 0; i<32; i++)
	{
		for(j=0;j<2500;j++){
			fprintf(fp1,"%f  ", Output[i][j]);
		}
			fprintf(fp1,"\n");
	}
    fclose(fp1);

	return 0;
}

