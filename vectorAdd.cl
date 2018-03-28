__kernel void vector_add(__global int *A, __global int *B, __global int *C)
{
    const int i = get_global_id(0);
    C[i] = A[i] + B[i];
}

__kernel void matrixMultiplication(__global float* inputA, __global float *inputB, __global float *outputC, int widthA, int heightA, int widthB, int heightB)
{ 
	int row = get_global_id(1);
	int col = get_global_id(0);

	float sum = 0.0f;

	for(int i=0; i<widthA; i++)
	{
		sum+=inputA[row*widthA+i] * inputB[i*widthB+col];
	}
	outputC[row*widthB+col]=sum;
}