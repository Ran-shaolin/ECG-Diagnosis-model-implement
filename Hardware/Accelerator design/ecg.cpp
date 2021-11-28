/*
Authored by Ran Shaolin
*/

#include "ecg.h"
#include <cstring>

using namespace std;

void clamp(int output_buffer_tmp[Tm][Tr],dtype output_buffer[Tm][Tr],m_dtype M_tmp[Tm])
{

	int acc_tmp=0;

	for(int i=0;i<Tm;i++){
		for(int j=0;j<Tr;j++){
			acc_tmp = output_buffer_tmp[i][j]*M_tmp[i];
			if (acc_tmp>255){
				acc_tmp = 255;
			}
			if (acc_tmp<0){
				acc_tmp = 0;
			}
			output_buffer[i][j]= acc_tmp;
		}
	}



}

void Input_to_buffer(dtype input_buf[Tn][OnChipInBuf_Height], dtype *input, int input_row, int tmp_r, int tmp_n, int Kernel_stride, int TN_MIN)//tmp为分块循环中的变量
{
	
	for(int i= 0;i<TN_MIN;i++)
	{
		memcpy(input_buf+i,input + i*input_row + tmp_r*Kernel_stride+ tmp_n*input_row, OnChipInBuf_Height*sizeof(dtype));
	}
}

void Weight_to_buffer(dtype weight_buf[Tm][Tn][Kernel_size], dtype *weight, int tmp_m, int tmp_n,int weight_row,int TM_MIN)//weight_sum是Tn*Kernel_size//weight_row是总的权重的输入通道乘以核尺寸
{
	
	dtype W_buffer_tmp[Tm][Tn*Kernel_size]={0};
	
	int weight_sum = Tn*Kernel_size;    
	
	for(int i= 0;i<Tm;i++)
	{
		memcpy(W_buffer_tmp + i, weight + i*weight_row + tmp_n*Kernel_size + tmp_m*weight_row, weight_sum*sizeof(dtype));
	}
	
	for(int i =0;i<Tm;i++){
		for(int j =0;j<Tn;j++){
			for(int l =0;l<Kernel_size;l++){
				weight_buf[i][j][l] = W_buffer_tmp[i][j*Kernel_size+l];
			}
		}
	}
	
}


void compute(dtype input_buffer[Tn][OnChipInBuf_Height], int output_buffer_tmp[Tm][Tr], dtype weight_buffer[Tm][Tn][Kernel_size],
			int bias_buffer[MAX_BIAS_LENGTH], dtype zero_p_buffer[MAX_BIAS_LENGTH], const int Kernel_stride,int TR_MIN,int TM_MIN, int TN_MIN,int tmp_m, int n,int nLoops)
{

	int local_bias_buffer[Tm] = { 0 };
#pragma HLS ARRAY_PARTITION variable=local_bias_buffer complete dim=1

	for(int i =0;i<Tm;i++){
		local_bias_buffer[i]=bias_buffer[i+tmp_m];
	}

	dtype local_zero_p_buffer[Tm] = { 0 };
#pragma HLS ARRAY_PARTITION variable=local_zero_p_buffer complete dim=1

	for(int i =0;i<Tm;i++){
		local_zero_p_buffer[i]=zero_p_buffer[i+tmp_m];
	}


	int partial_mul[Tn] = { 0 };
	
	int tmp_add1, tmp_add2, tmp_add3,tmp_add4,tmp_add5, tmp_add6, tmp_add7,tmp_add8;
	int tmp_add12,tmp_add34,tmp_add56,tmp_add78;
	int tmp_add1234,tmp_add5678;
	int tmp_add;

	for(int i = 0;i < Kernel_size; i++){
		for(int tr = 0;tr < TR_MIN;tr++){
			for (int tm = 0; tm < TM_MIN; tm++)
				{
#pragma HLS PIPELINE
					if (n == 0){
						output_buffer_tmp[tm][tr] = 0;
					}

					else{
						if(TN_MIN == Tn){
							for (int cycle = 0; cycle < 2; cycle++)
							{
						#pragma HLS UNROLL
								partial_mul[0+16*cycle]=(weight_buffer[tm][0+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[0+16*cycle][tr*Kernel_stride + i];
								partial_mul[1+16*cycle]=(weight_buffer[tm][1+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[1+16*cycle][tr*Kernel_stride + i];
								partial_mul[2+16*cycle]=(weight_buffer[tm][2+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[2+16*cycle][tr*Kernel_stride + i];
								partial_mul[3+16*cycle]=(weight_buffer[tm][3+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[3+16*cycle][tr*Kernel_stride + i];
								partial_mul[4+16*cycle]=(weight_buffer[tm][4+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[4+16*cycle][tr*Kernel_stride + i];
								partial_mul[5+16*cycle]=(weight_buffer[tm][5+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[5+16*cycle][tr*Kernel_stride + i];
								partial_mul[6+16*cycle]=(weight_buffer[tm][6+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[6+16*cycle][tr*Kernel_stride + i];
								partial_mul[7+16*cycle]=(weight_buffer[tm][7+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[7+16*cycle][tr*Kernel_stride + i];
								partial_mul[8+16*cycle]=(weight_buffer[tm][8+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[8+16*cycle][tr*Kernel_stride + i];
								partial_mul[9+16*cycle]=(weight_buffer[tm][9+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[9+16*cycle][tr*Kernel_stride + i];
								partial_mul[10+16*cycle]=(weight_buffer[tm][10+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[10+16*cycle][tr*Kernel_stride + i];
								partial_mul[11+16*cycle]=(weight_buffer[tm][11+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[11+16*cycle][tr*Kernel_stride + i];
								partial_mul[12+16*cycle]=(weight_buffer[tm][12+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[12+16*cycle][tr*Kernel_stride + i];
								partial_mul[13+16*cycle]=(weight_buffer[tm][13+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[13+16*cycle][tr*Kernel_stride + i];
								partial_mul[14+16*cycle]=(weight_buffer[tm][14+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[14+16*cycle][tr*Kernel_stride + i];
								partial_mul[15+16*cycle]=(weight_buffer[tm][15+16*cycle][i]-local_zero_p_buffer[tm]) * input_buffer[15+16*cycle][tr*Kernel_stride + i];

								tmp_add1 = partial_mul[0+16*cycle]+partial_mul[1+16*cycle];
								tmp_add2 = partial_mul[2+16*cycle]+partial_mul[3+16*cycle];
								tmp_add3 = partial_mul[4+16*cycle]+partial_mul[5+16*cycle];
								tmp_add4 = partial_mul[6+16*cycle]+partial_mul[7+16*cycle];
								tmp_add5 = partial_mul[8+16*cycle]+partial_mul[9+16*cycle];
								tmp_add6 = partial_mul[10+16*cycle]+partial_mul[11+16*cycle];
								tmp_add7 = partial_mul[12+16*cycle]+partial_mul[13+16*cycle];
								tmp_add8 = partial_mul[14+16*cycle]+partial_mul[15+16*cycle];

								tmp_add12 = tmp_add1+tmp_add2;
								tmp_add34 = tmp_add3+tmp_add4;
								tmp_add56 = tmp_add5+tmp_add6;
								tmp_add78 = tmp_add7+tmp_add8;

								tmp_add1234 = tmp_add12+tmp_add34;
								tmp_add5678 = tmp_add56+tmp_add78;

								tmp_add=tmp_add1234+tmp_add5678;

								output_buffer_tmp[tm][tr] += tmp_add;
							}
							if (i == 0 && n == 1)
							{
								output_buffer_tmp[tm][tr] += local_bias_buffer[tm];
							}

						}
						else{
							for (int tn = 0; tn < TN_MIN; tn++)
							{

								output_buffer_tmp[tm][tr] += (weight_buffer[tm][tn][i]-local_zero_p_buffer[tm]) * input_buffer[tn][tr*Kernel_stride + i];


							}
						}

					}

				}
		}
	}
}


void buffer_to_output(dtype out_buf[Tm][Tr], dtype *output, int output_row, int tmp_r, int tmp_m, int TM_MIN,int TR_MIN)//对应大写的TMP_M。out是输出特征图的首元素的地址
{
	
	for (int i = 0; i<TM_MIN; i++)
	{
		memcpy(output + i*output_row + tmp_r+ tmp_m*output_row,  out_buf+i, TR_MIN*sizeof(dtype));
	}
}


void copy_input_weight(dtype *input, dtype *weight, dtype input_buf[Tn][OnChipInBuf_Height], dtype weight_buf[Tm][Tn][Kernel_size],
						int input_row, int tmp_r, int tmp_n, int tmp_m, int Kernel_stride, int weight_row,int TN_MIN,int TM_MIN)
{
	Input_to_buffer(input_buf, input, input_row, tmp_r, tmp_n, Kernel_stride,TN_MIN);
	
	Weight_to_buffer(weight_buf, weight, tmp_m, tmp_n, weight_row,TM_MIN);

}


void pingpong_wrap(dtype *input, dtype *weight, int output_buffer_tmp[Tm][Tr], int bias_buffer[MAX_BIAS_LENGTH],dtype zero_p_buffer[MAX_BIAS_LENGTH],
							dtype input_buf0[Tn][OnChipInBuf_Height],dtype input_buf1[Tn][OnChipInBuf_Height],int input_row, 
							int tmp_r, int tmp_m, int Kernel_stride, int TR_MIN, int nLoops, int TM_MIN, const int InFM_num, int weight_row)
{
	dtype weight_buffer0[Tm][Tn][Kernel_size] = { 0 };

#pragma HLS ARRAY_PARTITION variable=weight_buffer0 complete dim=2

	dtype weight_buffer1[Tm][Tn][Kernel_size] = { 0 };

#pragma HLS ARRAY_PARTITION variable=weight_buffer1 complete dim=2

	bool pingpong = 0;
	int flag = 0;
	int TMP_N, TN_MIN, n;

	for (TMP_N = 0, n = 0; n < nLoops+1; n++, TMP_N += Tn)
	{
		if (n == nLoops){
			TMP_N = TMP_N - Tn;			
			TN_MIN = MIN(Tn, InFM_num - TMP_N);
		}
		else
		{
			TN_MIN= Tn;
		}

		if(pingpong == 0)
		{	
			copy_input_weight(input, weight, input_buf1, weight_buffer1, input_row, tmp_r, TMP_N, tmp_m, Kernel_stride, weight_row,TN_MIN,TM_MIN);
			
			compute(input_buf0,output_buffer_tmp, weight_buffer0, bias_buffer, zero_p_buffer, Kernel_stride, TR_MIN, TM_MIN, TN_MIN,tmp_m, n, nLoops);

			pingpong = 1;


		}
		else
		{
			copy_input_weight(input, weight, input_buf0, weight_buffer0, input_row, tmp_r, TMP_N, tmp_m, Kernel_stride, weight_row,TN_MIN,TM_MIN);

			compute(input_buf1, output_buffer_tmp, weight_buffer1, bias_buffer,zero_p_buffer, Kernel_stride, TR_MIN, TM_MIN, TN_MIN, tmp_m, n, nLoops);

			pingpong = 0;

		}
	}

}

void ECG_Stage( dtype *input, dtype *output, dtype *weight, int *bias, dtype *zero_p, m_dtype *M, const int Kernel_stride, const int InFM_num, const int OutFM_num,
				const int input_row, const int output_h,
				int mLoops, const int nLoops, const int rLoops, int weight_row)//output_h与output_row一样
				
{	
#pragma HLS INTERFACE m_axi depth=60000 port=input   offset=slave 
#pragma HLS INTERFACE m_axi depth=80000 port=output offset=slave 
#pragma HLS INTERFACE m_axi depth=12800 port=weight  offset=slave 
#pragma HLS INTERFACE m_axi depth=250 port=bias    offset=slave
#pragma HLS INTERFACE m_axi depth=250 port=zero_p    offset=slave
#pragma HLS INTERFACE m_axi depth=250 port=M    offset=slave

#pragma HLS INTERFACE s_axilite register port=return
#pragma HLS INTERFACE s_axilite register port=Kernel_stride
#pragma HLS INTERFACE s_axilite register port=InFM_num
#pragma HLS INTERFACE s_axilite register port=OutFM_num
#pragma HLS INTERFACE s_axilite register port=input_row
#pragma HLS INTERFACE s_axilite register port=output_h
#pragma HLS INTERFACE s_axilite register port=mLoops
#pragma HLS INTERFACE s_axilite register port=nLoops
#pragma HLS INTERFACE s_axilite register port=rLoops
#pragma HLS INTERFACE s_axilite register port=weight_row


	dtype input_buffer0[Tn][OnChipInBuf_Height] = { 0 };
#pragma HLS ARRAY_PARTITION variable=input_buffer0 complete dim=1


	dtype input_buffer1[Tn][OnChipInBuf_Height] = { 0 };
#pragma HLS ARRAY_PARTITION variable=input_buffer1 complete dim=1

	int output_buffer_tmp[Tm][Tr] = { 0 };
#pragma HLS ARRAY_PARTITION variable=output_buffer_tmp complete dim=1

	dtype output_buffer1[Tm][Tr] = { 0 };
#pragma HLS ARRAY_PARTITION variable=output_buffer1 complete dim=1

	int bias_buffer[MAX_BIAS_LENGTH] = { 0 };

	dtype zero_p_buffer[MAX_BIAS_LENGTH] = {0};


	m_dtype M_buffer[MAX_BIAS_LENGTH] = {0};

	m_dtype M_tmp[Tm]={0};

	int r, c, m;
	int TMP_R, TMP_M,TMP_N;
	int TR_MIN,TM_MIN;
	
	bool pingpongm;
	
	memcpy(bias_buffer,bias,OutFM_num*sizeof(int));
	
	memcpy(zero_p_buffer ,zero_p, OutFM_num*sizeof(dtype));

	memcpy(M_buffer,M,OutFM_num*sizeof(int));

	for(TMP_R = 0,r = 0; r < rLoops; r++, TMP_R += Tr)
	{
		TR_MIN = MIN(Tr,output_h -TMP_R);
	  	
		pingpongm = 0;
		for(TMP_M = 0, m = 0; m < mLoops; m++,TMP_M += Tm)
		{
			
			TM_MIN = MIN(Tm,OutFM_num-TMP_M);

			for(int i=0;i<TM_MIN;i++){
				M_tmp[i]= M_buffer[m*Tm+i];
			}

			if(pingpongm==0)
				{
					pingpong_wrap(input, weight, output_buffer_tmp, bias_buffer, zero_p_buffer,
											input_buffer0, input_buffer1, input_row,
											TMP_R, TMP_M, Kernel_stride, TR_MIN, nLoops, TM_MIN, InFM_num, weight_row);				
					
					clamp(output_buffer_tmp,output_buffer1,M_tmp);//

					buffer_to_output(output_buffer1, output, output_h, TMP_R, TMP_M, TM_MIN,TR_MIN);
					memset(output_buffer1,0,sizeof(dtype)*Tm*Tr);
				
					pingpongm = 1;
				}
			
				{					
					pingpong_wrap(input, weight, output_buffer0, bias_buffer,
											input_buffer0, input_buffer1, input_row,
											TMP_R, TMP_M, Kernel_stride, TR_MIN, nLoops, TM_MIN, InFM_num, weight_row);
					
					buffer_to_output(output_buffer1, output, output_h, TMP_R, TMP_M);
					
					pingpongm = 0;
				}
		}
	}
}
		
