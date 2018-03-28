__kernel void image_rotate(__global float *destination, __global float *source, int w, int h, float sinTheta, float cosTheta)
{
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);

	float xpos = ((float)ix) * cosTheta + ((float)iy)*sinTheta;
	float ypos = -1.0*((float)ix) * sinTheta + ((float)iy)*cosTheta;

	if( ( (int) xpos >= 0) && ( (int) xpos < w) && ( (int) ypos >= 0) && ( (int) ypos < h) )
	{
		destination[(int)ypos*w + (int)xpos] = source[iy*w+ix];
	}
}


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void convolution(__read_only image2d_t image, __constant float *mask, __global float *blurredImage, __private int maskSize)
{
	/*get position x, y*/
	const int2 pos = {get_global_id(0), get_global_id(1) }; 

	/*collect neighbor values*/
	float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            sum += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)]
                *read_imagef(image, sampler, pos + (int2)(a,b)).x;
        }
    }
 
    blurredImage[pos.x+pos.y*get_global_size(0)] = sum;
}