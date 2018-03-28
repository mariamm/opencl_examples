// OpenCL ray tracing tutorial by Sam Lapere, 2016
// http://raytracey.blogspot.com
#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <fstream>
#include <vector>
#include <CL\cl.hpp>
#include <opencv2\opencv.hpp>
#pragma comment(lib, "OpenCL.lib")

const int image_width = 1280;
const int image_height = 720;

cl_float4* cpu_output;
cl::CommandQueue m_queue;
cl::Kernel m_kernel;
cl::Context m_context;
cl::Program m_program;
cl::Buffer cl_output;

void pickPlatform(cl::Platform& platform, const std::vector<cl::Platform>& platforms){
	
	if (platforms.size() == 1) platform = platforms[0];
	else{
		int input = 0;
        std::cout << "\nChoose an OpenCL platform: ";
        std::cin >> input;

		// handle incorrect user input
		while (input < 1 || input > platforms.size()){
            std::cin.clear(); //clear errors/bad flags on cin
            std::cin.ignore(std::cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
            std::cout << "No such option. Choose an OpenCL platform: ";
            std::cin >> input;
		}
		platform = platforms[input - 1];
	}
}

void pickDevice(cl::Device& device, const std::vector<cl::Device>& devices){
	
	if (devices.size() == 1) device = devices[0];
	else{
		int input = 0;
        std::cout << "\nChoose an OpenCL device: ";
        std::cin >> input;

		// handle incorrect user input
		while (input < 1 || input >  devices.size()){
            std::cin.clear(); //clear errors/bad flags on cin
            std::cin.ignore(std::cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
            std::cout << "No such option. Choose an OpenCL device: ";
            std::cin >> input;
		}
		device = devices[input - 1];
	}
}

void printErrorLog(const cl::Program& m_program, const cl::Device& device){
	
	// Get the error log and print to console
    std::string buildlog = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    std::cerr << "Build log:" << std::endl << buildlog << std::endl;

	system("PAUSE");
	exit(1);
}

void selectRenderMode(unsigned int& rendermode){
	std::cout << std::endl << "Rendermodes: " << std::endl << std::endl;
	std::cout << "\t(1) Simple gradient" <<std::endl;
	std::cout << "\t(2) Sphere with plain colour" <<std::endl;
	std::cout << "\t(3) Sphere with cosine weighted colour" <<std::endl;
	std::cout << "\t(4) Stripey sphere" <<std::endl;
	std::cout << "\t(5) Sphere with screen door effect" <<std::endl;
	std::cout << "\t(6) Sphere with normals" <<std::endl;

	unsigned int input;
	std::cout <<std::endl << "Select rendermode (1-6): ";
	std::cin >> input; 

	// handle incorrect user input
	while (input < 1 || input > 6){
		std::cin.clear(); //clear errors/bad flags on cin
		std::cin.ignore(std::cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
		std::cout << "No such option. Select rendermode: ";
        std::cin >> input;
	}
	rendermode = input;
}

void initOpenCL(std::string kernelFileName, const char* kernelName)
{
	// Get all available OpenCL platforms (e.g. AMD OpenCL, Nvidia CUDA, Intel OpenCL)
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::cout << "Available OpenCL platforms : " << std::endl << std::endl;
	for (int i = 0; i <  platforms.size(); i++)
       std::cout << "\t" << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;

	// Pick one platform
    cl::Platform platform = platforms[2];
	//pickPlatform(platform, platforms);
    std::cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	// Get available OpenCL devices on platform
    std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    std::cout << "Available OpenCL devices on this platform: " << std::endl << std::endl;
	for (int i = 0; i <  devices.size(); i++){
		std::cout << "\t" << i + 1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\t\tMax compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
		std::cout << "\t\tMax work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl << std::endl;
	}

	// Pick one device
    cl::Device device = devices[0];
	//pickDevice(device, devices);
	std::cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() <<std::endl;
	std::cout << "\t\t\tMax compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() <<std::endl;
	std::cout << "\t\t\tMax work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() <<std::endl;

	// Create an OpenCL context and command queue on that device.
	m_context = cl::Context(device);
	m_queue = cl::CommandQueue(m_context, device);

	// Convert the OpenCL source code to a string
    std::string source;
    std::ifstream file(kernelFileName);
	if (!file){
        std::cout << "\nNo OpenCL file found!" <<std::endl << "Exiting..." << std::endl;
		system("PAUSE");
		exit(1);
	}
	while (!file.eof()){
		char line[256];
		file.getline(line, 255);
		source += line;
	}

	const char* kernel_source = source.c_str();

	// Create an OpenCL program by performing runtime source compilation for the chosen device
	m_program = cl::Program(m_context, kernel_source);
	cl_int result = m_program.build({ device });
	if (result) std::cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << std::endl;
	if (result == CL_BUILD_PROGRAM_FAILURE) printErrorLog(m_program, device);

	// Create a kernel (entry point in the OpenCL source program)
	//kernel = cl::Kernel(program, "render_kernel");
    m_kernel = cl::Kernel(m_program, kernelName, &result);
}

inline float clamp(float x){ return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

// convert RGB float in range [0,1] to int in range [0, 255]
inline int toInt(float x){ return int(clamp(x) * 255 + .5); }

void saveImage()
{
    cv::Mat image = cv::Mat::zeros(image_height, image_width, CV_8UC3);

    int idx = 0;
	// loop over pixels, write RGB values
    for (int i = 0; i < image.rows; i++)
    {
        uchar *pRowImage = image.ptr<uchar>(i);
        for (int j = 0; j < image.cols; j++)
        {
            cv::Vec3b* ptr = reinterpret_cast<cv::Vec3b*>(pRowImage);
            ptr[j][0] = toInt(cpu_output[idx].s[2]);
            ptr[j][1] = toInt(cpu_output[idx].s[1]);
            ptr[j][2] = toInt(cpu_output[idx].s[0]);
            idx++;
        }
    }

    cv::imwrite("image.png", image);
}

void cleanUp()
{
	delete cpu_output;
}



void pathTracingTest()
{

    // allocate memory on CPU to hold image
    cpu_output = new cl_float3[image_width * image_height];

    // Create image buffer on the OpenCL device
    cl_output = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, image_width * image_height * sizeof(cl_float3));

    // pick a rendermode
    unsigned int rendermode;
    selectRenderMode(rendermode);

    // specify OpenCL kernel arguments
    m_kernel.setArg(0, cl_output);
    m_kernel.setArg(1, image_width);
    m_kernel.setArg(2, image_height);
    m_kernel.setArg(3, rendermode);

    // every pixel in the image has its own thread or "work item",
    // so the total amount of work items equals the number of pixels
    std::size_t global_work_size = image_width * image_height;
    std::size_t local_work_size = 256;

    // launch the kernel
    m_queue.enqueueNDRangeKernel(m_kernel, NULL, global_work_size, local_work_size);
    m_queue.finish();

    // read and copy OpenCL output to CPU
    m_queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, image_width * image_height * sizeof(cl_float3), cpu_output);

    // save image to PPM format
    saveImage();
    std::cout << "Rendering done!\nSaved image'" << std::endl;

    // release memory
    cleanUp();

    system("PAUSE");
}
void vectorAddTest()
{
    //create vector inputs
    const int vectorSize = 1379483392;
    const int vectorSizeMemory = (vectorSize) * sizeof(int);
    int *A = new int[vectorSize];
    int *B = new int[vectorSize];
    int *C = new int[vectorSize];

    for (int i = 0; i < vectorSize; i++)
    {
        A[i] = i;
        B[i] = vectorSize - i;
    }


    std::clock_t start;
    double duration;
    start = std::clock();
    //create memory buffers
    cl::Buffer memoryA = cl::Buffer(m_context, CL_MEM_READ_ONLY, vectorSizeMemory);
    cl::Buffer memoryB = cl::Buffer(m_context, CL_MEM_READ_ONLY, vectorSizeMemory);
    cl::Buffer memoryC = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, vectorSizeMemory);
    //copy arrays to buffers
    m_queue.enqueueWriteBuffer(memoryA, CL_TRUE, 0, vectorSizeMemory, A);
    m_queue.enqueueWriteBuffer(memoryB, CL_TRUE, 0, vectorSizeMemory, B);

    //set the kernel argumnet
    m_kernel.setArg(0, memoryA);
    m_kernel.setArg(1, memoryB);
    m_kernel.setArg(2, memoryC);


    std::clock_t startkernel;
    double durationkernel;
    startkernel = std::clock();
    //launch the kernel    
    std::size_t global_work_size = vectorSize;
    std::size_t local_work_size = 256;
    m_queue.enqueueNDRangeKernel(m_kernel, NULL, global_work_size, local_work_size);
    m_queue.finish();
    m_queue.enqueueReadBuffer(memoryC, CL_TRUE, 0, vectorSizeMemory, C);
    durationkernel = (std::clock() - startkernel) / (double)CLOCKS_PER_SEC;
    std::cout << "Duration kernel only: " << durationkernel << std::endl;

    //clean up
    //delete[]A;
    //delete[]B;
    //delete[]C;
    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "Duration total GPU: " << duration << std::endl;


    std::clock_t start2;
    start2 = std::clock();

    for (int i = 0; i < vectorSize; i++)
    {
        C[i] = A[i] + B[i];
    }


    duration = (std::clock() - start2) / (double)CLOCKS_PER_SEC;
    std::cout << "Duration loop cpu: " << duration << std::endl;

    delete[]A;
    delete[]B;
    delete[]C;
    system("PAUSE");

}

float * createGaussianMask(float sigma, int  maskSize) 
{
    float * mask = new float[(maskSize * 2 + 1)*(maskSize * 2 + 1)];
    float sum = 0.0f;
    for (int a = -maskSize; a < maskSize + 1; a++) 
    {
        for (int b = -maskSize; b < maskSize + 1; b++) 
        {
            float temp = exp(-((float)(a*a + b*b) / (2 * sigma*sigma)));
            sum += temp;
            mask[a + maskSize + (b + maskSize)*(maskSize * 2 + 1)] = temp;
        }
    }
    // Normalize the mask
    for (int i = 0; i < (maskSize * 2 + 1)*(maskSize * 2 + 1); i++)
        mask[i] = mask[i] / sum;


    return mask;
}

void blurTest(std::string type, int maskSize, float sigma = 0)
{
    float * mask = new float[maskSize];
    // Create Gaussian mask
    if(type == "gaussian")
    { 
        mask = createGaussianMask(sigma, maskSize);
    }
    else if (type == "average")
    {
        float value = 1.f / maskSize;
        for (int i = 0; i < maskSize; i++)
        {
            mask[i] = value;
        }
    }
    else
    {
        return;
    }
    cl_int err;
    cv::Mat image = cv::imread("1.jpg");
    cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32FC3, 1/255.0);
    cl::Image2D clImage = cl::Image2D(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_FLOAT), imageFloat.cols, imageFloat.rows, 0, (void*)imageFloat.data, &err);

    //create buffers
    cl::Buffer clMask = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(maskSize * 2 + 1)*(maskSize * 2 + 1), mask);
    cl::Buffer clResult = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, sizeof(float*)*imageFloat.cols*imageFloat.rows, NULL, &err);
    //set kernel arguments
    m_kernel.setArg(0, clImage);
    m_kernel.setArg(1, clMask);
    m_kernel.setArg(2, clResult);
    m_kernel.setArg(3, maskSize);

    //launch kernel
    m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, cl::NDRange(imageFloat.cols, imageFloat.rows), cl::NullRange);

    //get results from device
    float* data = new float[imageFloat.cols*imageFloat.rows];
    m_queue.enqueueReadBuffer(clResult, CL_TRUE, 0, sizeof(float)*imageFloat.cols*imageFloat.rows, data);


    cv::Mat blurredImage = cv::Mat(imageFloat.rows, imageFloat.cols, CV_32F, data);
    cv::imwrite(type+"-blurred-image.png", blurredImage);

}

void matrixMulTest()
{
    int widthA = 3;
    int heightA = 2;

    int widthB = heightA;
    int heightB = widthA;

    float* matA = new float[widthA*heightA];
    float* matB = new float[widthB*heightB];

    for (int i = 0; i < widthA*heightA; i++)
    {
        matA[i] = i;
        matB[i] = i;
    }
    float *matC = new float[widthA*widthA];

    cl::Buffer memoryA = cl::Buffer(m_context, CL_MEM_READ_ONLY, widthA*heightB);
    cl::Buffer memoryB = cl::Buffer(m_context, CL_MEM_READ_ONLY, widthA*heightB);
    cl::Buffer memoryC = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, widthA*widthA);
    //copy arrays to buffers
    m_queue.enqueueWriteBuffer(memoryA, CL_TRUE, 0, widthA*heightB, matA);
    m_queue.enqueueWriteBuffer(memoryB, CL_TRUE, 0, widthA*heightB, matB);

    //set the kernel argumnet
    m_kernel.setArg(0, memoryA);
    m_kernel.setArg(1, memoryB);
    m_kernel.setArg(2, memoryC);
    m_kernel.setArg(3, widthA);
    m_kernel.setArg(4, heightA);
    m_kernel.setArg(5, widthB);
    m_kernel.setArg(6, heightB);


    //launch the kernel    
    std::size_t global_work_size = widthA*widthA;
    std::size_t local_work_size = 256;
    m_queue.enqueueNDRangeKernel(m_kernel, NULL, global_work_size, local_work_size);
    m_queue.finish();
    m_queue.enqueueReadBuffer(memoryC, CL_TRUE, 0, widthA*widthA, (void*)matC);
    
    for (int i = 0; i < widthA*widthA; i++)
    {
        std::cout << matC[i] << " | ";
        if ((i + 1) % 3 == 0)
        {
            std::cout << std::endl;
        }
    }
    std::cout << "Done "  << std::endl;

    delete[]matA;
    delete[]matB;
    delete[]matC;
    system("PAUSE");
}

void rotateImageTest()
{
    cl_int err;
    cv::Mat image = cv::imread("1.jpg");
    cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32FC1, 1 / 255.0);

    float *imageArray = imageFloat.ptr<float>(0);

    int width = image.cols;
    int height = image.rows;
    float cos_theta = cosf(0.785398f);
    float sin_theta = sinf(0.785398f);
    //create Buffers

    cl::Buffer clImage = cl::Buffer(m_context, CL_MEM_READ_ONLY, width*height * sizeof(float));
    cl_output = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, width*height * sizeof(float), NULL, &err);

    m_queue.enqueueWriteBuffer(clImage, CL_TRUE, 0, width*height * sizeof(float), imageArray);
    //set kernel arguments
    m_kernel.setArg(0, cl_output);
    m_kernel.setArg(1, clImage);
    m_kernel.setArg(2, width);
    m_kernel.setArg(3, height);
    m_kernel.setArg(4, cos_theta);
    m_kernel.setArg(5, sin_theta);

    //launch kernel
    m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, cl::NDRange(imageFloat.cols, imageFloat.rows), cl::NullRange);

    //get results from device
    float* data = new float[imageFloat.cols*imageFloat.rows];
    m_queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, sizeof(float)*imageFloat.cols*imageFloat.rows, data);


    cv::Mat rotatedImage = cv::Mat(imageFloat.rows, imageFloat.cols, CV_32F, data);
    cv::imwrite("rotated-image.png", rotatedImage);


}
void main() {

    // initialise OpenCL
    initOpenCL("advanced_kernels.cl", "image_rotate");

   // pathTracingTest();

    //vectorAddTest();
    //blurTest("gaussian", 30, 5.5f);
    //matrixMulTest();
    rotateImageTest();
}