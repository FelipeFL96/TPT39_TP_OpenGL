#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <CL/cl.h>
#include <CL/cl_ext.h>
///#include <chrono>

using namespace cv;
using namespace std;
#define STRING_BUFFER_LEN 1024
#define SHOW
#define TOTAL_FRAMES 300


void callback(const char *buffer, size_t length, size_t final, void *user_data) {
    fwrite(buffer, 1, length, stdout);
}

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

void checkError(int status, const char *msg) {
	if(status != CL_SUCCESS)	
		printf("%s: [%d] %s\n", msg, status, getErrorString(status));
}

void print_clbuild_errors(cl_program program,cl_device_id device) {
    cout << "Program Build failed\n";
    size_t length;
    char buffer[2048];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
    cout << "--- Build log ---\n " << buffer << endl;
    exit(1);
}

unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file: %s\n",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  
  return output;
}

void initOpenCL(cl_context& context, cl_command_queue& queue, cl_program& program, cl_kernel& kernel) {
    char char_buffer[STRING_BUFFER_LEN];
   
    cl_platform_id platform;
    cl_device_id device;
    cl_context_properties context_properties[] =
   
    { 
        CL_CONTEXT_PLATFORM, 0,
        CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
        CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
        0
    };

    clGetPlatformIDs(2, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    
    context_properties[1] = (cl_context_properties)platform;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    //Reading nd compiling kernel program

    unsigned char **opencl_program = read_file("videofilter.cl");
    
    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
    if (program == NULL) {
        printf("Program creation failed\n");
	}	
    
    int success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (success != CL_SUCCESS)
        print_clbuild_errors(program,device);

    kernel = clCreateKernel(program, "videofilter", NULL);
    clEnqueueTask(queue, kernel, 0, NULL, NULL);
}

void endOpenCL(cl_context& context, cl_command_queue& queue, cl_program& program,  cl_kernel& kernel) {
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void gpuVideoFilter(cl_context context, cl_command_queue queue, cl_kernel kernel, VideoCapture &inputVideo, VideoWriter &outputVideo, Size frameSize) {
    cl_mem input_kernel_buffer;
    cl_mem input_frame_buffer;
    cl_mem output_frame_buffer;
    int status;
    cl_int errcode;

    // Execution events
    cl_event write_event[2];
    cl_event kernel_event;

    // Gaussian Kernel
    //Mat gaussianKernel = getGaussianKernel(3, 1) * getGaussianKernel(3, 1).t();
    ///gaussianKernel.convertTo(gaussianKernel, CV_32FC1);

    // I/O buffers.
    input_kernel_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 9*sizeof(float), NULL, &status);
    input_frame_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, (frameSize.width * frameSize.height)*sizeof(char), NULL, &status);
    output_frame_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (frameSize.width * frameSize.height)*sizeof(char), NULL, &status);

    // Kernel arguments
    unsigned argi = 0;
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_kernel_buffer);
    checkError(status, "Failed to set argument 0");
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_frame_buffer);
    checkError(status, "Failed to set argument 1");
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_frame_buffer);
    checkError(status, "Failed to set argument 2");

    for (int frame = 0; frame < TOTAL_FRAMES; frame++) {
        Mat inputVideoFrame, displayFrame;
        Mat grayFrame, edge_x, edge_y, edge, edge_inv;

        inputVideo >> inputVideoFrame;
        cvtColor(inputVideoFrame, grayFrame, CV_BGR2GRAY);


        // Mapping the GPU input buffers
        float *input_kernel = (float*) clEnqueueMapBuffer(queue, input_kernel_buffer, CL_TRUE,
         CL_MAP_WRITE, 0, 9*sizeof(float), 0, NULL, &write_event[0], &errcode);
        checkError(errcode, "Error");
        unsigned char *input_frame = (unsigned char *) clEnqueueMapBuffer(queue, input_frame_buffer, CL_TRUE,
         CL_MAP_WRITE, 0, (frameSize.width * frameSize.height)*sizeof(char), 0, NULL, &write_event[1], &errcode);
        checkError(errcode, "Error");
        
        // Filling bufers with data
        input_kernel[0] = 3;//(float) 1/16;
        input_kernel[1] = 0;//(float) 2/16;
        input_kernel[2] = -3;//(float) 1/16;
        input_kernel[3] = 10;//(float) 2/16;
        input_kernel[4] = 0;//(float) 4/16;
        input_kernel[5] = -10;//(float) 2/16;
        input_kernel[6] = 3;//(float) 1/16;
        input_kernel[7] = 0;//(float) 2/16;
        input_kernel[8] = -3;//(float) 1/16;
        //memcpy(input_kernel, gaussianKernel.data, 9*sizeof(float));
        memcpy(input_frame, grayFrame.data, (frameSize.width * frameSize.height)*sizeof(char));

        clEnqueueUnmapMemObject(queue, input_kernel_buffer, input_kernel, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, input_frame_buffer, input_frame, 0, NULL, NULL);

        const unsigned int global_work_size[2] = {frameSize.width, frameSize.height} ;
        //Start time measurement (1 Frame)
        //auto frame_t1 = std::chrono::high_resolution_clock::now();
        status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
            global_work_size, NULL, 2, write_event, &kernel_event);
        clWaitForEvents(1, &kernel_event);
        //auto frame_t2 = std::chrono::high_resolution_clock::now();
        //Stop time measurement (1 Frame)

        // Mapping the GPU ouput buffer
        unsigned char *output_frame = (unsigned char *) clEnqueueMapBuffer(queue, output_frame_buffer, CL_TRUE,
         CL_MAP_WRITE, 0, (frameSize.width*frameSize.height)*sizeof(unsigned char), 0, NULL, NULL, &errcode);
        checkError(errcode, "Error");

        // Recovering data from GPU computation
        Mat outputVideoFrame(frameSize.height, frameSize.width, CV_8U, output_frame);
        memcpy(outputVideoFrame.data, output_frame, (frameSize.height * frameSize.width)*sizeof(unsigned char));

        clEnqueueUnmapMemObject(queue, output_frame_buffer, output_frame, 0, NULL, NULL);

        cvtColor(outputVideoFrame, displayFrame, CV_GRAY2BGR);
        outputVideo << displayFrame;

        //auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_t2 - frame_t1).count();
        //std::cout << "GPU frame time: " << cpu_duration << " ms" << std::endl;
    }
}

void disposeVideos(VideoCapture& inputVideo, VideoWriter& outputVideo1, VideoWriter& outputVideo2) {
	inputVideo.release();
    outputVideo1.release();
    outputVideo2.release();
}

int main(int, char**) {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Reading Input video
    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened()) {
        printf("Failed in reading video file\n");
        return EXIT_FAILURE;
    }
    Size frameSize = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	cout << "SIZE:" << frameSize << endl;


    // Preparing Output videos
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    	
    VideoWriter cpuOutputVideo, gpuOutputVideo;

    cpuOutputVideo.open("./cpu_output.avi", ex, 25, frameSize, true);
    if (!cpuOutputVideo.isOpened()) {
        cout  << "Could not open the CPU output video for write: " << endl;
        return EXIT_FAILURE;
    }
    gpuOutputVideo.open("./gpu_output.avi", ex, 25, frameSize, true);
    if (!gpuOutputVideo.isOpened()) {
        cout  << "Could not open the GPU output video for write: "  << endl;
        return EXIT_FAILURE;
    }

    // Generating context environment
    initOpenCL(context, queue, program, kernel);
    
    // GPU Processing
    gpuVideoFilter(context, queue, kernel, camera, gpuOutputVideo, frameSize);

    // Cleaning environment
    endOpenCL(context, queue, program, kernel);
    disposeVideos(camera, cpuOutputVideo, gpuOutputVideo);

    return EXIT_SUCCESS;
}