// cudnn.h — macOS arm64 cuDNN-compatible header for the oneDNN-backed shim.
//
// Declares the classic-cuDNN descriptor + op surface needed to LINK consumers
// (oneDNN's nvidia backend reference, MLX's direct C calls). Pairs with
// libcudnn.dylib, which backs softmax / activation / pooling / (scaffold) conv
// with REAL CPU compute via Intel oneDNN. Everything else returns
// CUDNN_STATUS_NOT_SUPPORTED — never a fake success.
//
// This is macOS-native CPU compute, NOT eGPU cuDNN. Real GPU cuDNN = sp3.
#ifndef MACOS_ONEDNN_CUDNN_H
#define MACOS_ONEDNN_CUDNN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- status ---
typedef enum {
  CUDNN_STATUS_SUCCESS = 0,
  CUDNN_STATUS_NOT_INITIALIZED = 1,
  CUDNN_STATUS_BAD_PARAM = 3,
  CUDNN_STATUS_INTERNAL_ERROR = 4,
  CUDNN_STATUS_NOT_SUPPORTED = 9
} cudnnStatus_t;

// --- data types ---
typedef enum {
  CUDNN_DATA_FLOAT = 0,
  CUDNN_DATA_DOUBLE = 1,
  CUDNN_DATA_HALF = 2,
  CUDNN_DATA_INT8 = 3,
  CUDNN_DATA_INT32 = 4,
  CUDNN_DATA_BFLOAT16 = 14
} cudnnDataType_t;

// --- tensor formats ---
typedef enum {
  CUDNN_TENSOR_NCHW = 0,
  CUDNN_TENSOR_NHWC = 1,
  CUDNN_TENSOR_NCHW_VECT_C = 2
} cudnnTensorFormat_t;

// --- nan propagation ---
typedef enum {
  CUDNN_NOT_PROPAGATE_NAN = 0,
  CUDNN_PROPAGATE_NAN = 1
} cudnnNanPropagation_t;

// --- softmax ---
typedef enum {
  CUDNN_SOFTMAX_FAST = 0,
  CUDNN_SOFTMAX_ACCURATE = 1,
  CUDNN_SOFTMAX_LOG = 2
} cudnnSoftmaxAlgorithm_t;
typedef enum {
  CUDNN_SOFTMAX_MODE_INSTANCE = 0,
  CUDNN_SOFTMAX_MODE_CHANNEL = 1
} cudnnSoftmaxMode_t;

// --- activation ---
typedef enum {
  CUDNN_ACTIVATION_SIGMOID = 0,
  CUDNN_ACTIVATION_RELU = 1,
  CUDNN_ACTIVATION_TANH = 2,
  CUDNN_ACTIVATION_CLIPPED_RELU = 3,
  CUDNN_ACTIVATION_ELU = 4,
  CUDNN_ACTIVATION_IDENTITY = 5
} cudnnActivationMode_t;

// --- pooling ---
typedef enum {
  CUDNN_POOLING_MAX = 0,
  CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
  CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
  CUDNN_POOLING_MAX_DETERMINISTIC = 3
} cudnnPoolingMode_t;

// --- convolution ---
typedef enum {
  CUDNN_CONVOLUTION = 0,
  CUDNN_CROSS_CORRELATION = 1
} cudnnConvolutionMode_t;
typedef enum {
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
} cudnnConvolutionFwdAlgo_t;

// --- opaque handles + descriptors ---
typedef struct cudnnContext* cudnnHandle_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct cudnnTensorStruct* cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct* cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct* cudnnConvolutionDescriptor_t;
typedef struct cudnnActivationStruct* cudnnActivationDescriptor_t;
typedef struct cudnnPoolingStruct* cudnnPoolingDescriptor_t;

// ===========================================================================
// Handle lifecycle + version
// ===========================================================================
cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t* streamId);
size_t cudnnGetVersion(void);
size_t cudnnGetCudartVersion(void);
const char* cudnnGetErrorString(cudnnStatus_t status);

// Diagnostic: proves the oneDNN backend linked (non-standard helper).
int cudnn_shim_onednn_version(void);

// ===========================================================================
// Tensor descriptor
// ===========================================================================
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* desc);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t desc);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t desc,
    cudnnTensorFormat_t format, cudnnDataType_t dataType,
    int n, int c, int h, int w);
cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t desc,
    cudnnDataType_t dataType, int n, int c, int h, int w,
    int nStride, int cStride, int hStride, int wStride);
cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t desc,
    cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]);
cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t desc,
    cudnnTensorFormat_t format, cudnnDataType_t dataType, int nbDims,
    const int dimA[]);
cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t desc,
    cudnnDataType_t* dataType, int* n, int* c, int* h, int* w,
    int* nStride, int* cStride, int* hStride, int* wStride);

// ===========================================================================
// Filter descriptor
// ===========================================================================
cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* desc);
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t desc);
cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t desc,
    cudnnDataType_t dataType, cudnnTensorFormat_t format,
    int k, int c, int h, int w);
cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t desc,
    cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims,
    const int filterDimA[]);

// ===========================================================================
// Convolution descriptor + forward (SCAFFOLD)
// ===========================================================================
cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* desc);
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t desc);
cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t desc,
    int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w,
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType);
cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t desc,
    int groupCount);
cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t desc,
    int mathType);
cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int* n, int* c, int* h, int* w);
cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle,
    int* count);
cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t* sizeInBytes);
cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle,
    const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x,
    const cudnnFilterDescriptor_t wDesc, const void* w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);

// ===========================================================================
// Activation descriptor + forward (REAL)
// ===========================================================================
cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* desc);
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t desc);
cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t desc,
    cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double coef);
cudnnStatus_t cudnnActivationForward(cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);

// ===========================================================================
// Softmax forward (REAL)
// ===========================================================================
cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);

// ===========================================================================
// Pooling descriptor + forward (REAL)
// ===========================================================================
cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* desc);
cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t desc);
cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t desc,
    cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt,
    int windowHeight, int windowWidth, int verticalPadding,
    int horizontalPadding, int verticalStride, int horizontalStride);
cudnnStatus_t cudnnGetPooling2dForwardOutputDim(
    const cudnnPoolingDescriptor_t poolingDesc,
    const cudnnTensorDescriptor_t inputDesc, int* n, int* c, int* h, int* w);
cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MACOS_ONEDNN_CUDNN_H
