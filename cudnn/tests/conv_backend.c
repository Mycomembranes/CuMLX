// conv_backend.c — exercise the ported cuDNN v9 BACKEND descriptor API directly:
// build a conv-forward op graph by hand, cudnnBackendExecute it, and verify the
// output against a hand-computed reference. (Validates the descriptor state
// machine + Execute + the oneDNN-CPU path, independent of cudnn-frontend.)
//
//   X = 1x1x3x3 [[1,2,3],[4,5,6],[7,8,9]]
//   W = 1x1x2x2 [[1,0],[0,1]]   (K=1,C=1,R=2,S=2)
//   stride 1, pad 0, dilation 1, cross-correlation
//   => Y = 1x1x2x2 [[6,8],[12,14]]
#include <cudnn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(call)                                                               \
  do {                                                                         \
    cudnnStatus_t _s = (call);                                                 \
    if (_s != CUDNN_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "FAIL %s -> %d\n", #call, (int)_s);                      \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static cudnnBackendDescriptor_t mk_tensor(int64_t uid, const int64_t *dims,
                                          const int64_t *strides) {
  cudnnBackendDescriptor_t t;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &t);
  cudnnDataType_t dt = CUDNN_DATA_FLOAT;
  int64_t nd = 4;
  cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
  cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, nd, dims);
  cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, nd, strides);
  cudnnBackendFinalize(t);
  return t;
}

int main(void) {
  cudnnHandle_t handle;
  cudnnCreate(&handle);

  int64_t xd[4] = {1, 1, 3, 3}, xs[4] = {9, 9, 3, 1};
  int64_t wd[4] = {1, 1, 2, 2}, ws[4] = {4, 4, 2, 1};
  int64_t yd[4] = {1, 1, 2, 2}, ys[4] = {4, 4, 2, 1};
  cudnnBackendDescriptor_t X = mk_tensor(1, xd, xs);
  cudnnBackendDescriptor_t W = mk_tensor(2, wd, ws);
  cudnnBackendDescriptor_t Y = mk_tensor(3, yd, ys);

  // convolution descriptor
  cudnnBackendDescriptor_t conv;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &conv);
  cudnnDataType_t comp = CUDNN_DATA_FLOAT;
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  int64_t sp = 2, fstride[2] = {1, 1}, pad[2] = {0, 0}, dil[2] = {1, 1};
  cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp);
  cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
  cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &sp);
  cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, fstride);
  cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pad);
  cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pad);
  cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dil);
  cudnnBackendFinalize(conv);

  // operation: conv forward
  cudnnBackendDescriptor_t op;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &op);
  double alpha = 1.0, beta = 0.0;
  cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &X);
  cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &W);
  cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &Y);
  cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv);
  cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_DOUBLE, 1, &alpha);
  cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_DOUBLE, 1, &beta);
  cudnnBackendFinalize(op);

  // operation graph
  cudnnBackendDescriptor_t graph;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph);
  cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
  cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op);
  cudnnBackendFinalize(graph);

  // engine -> enginecfg -> execution plan
  cudnnBackendDescriptor_t engine;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine);
  int64_t gidx = 0;
  cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
  cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gidx);
  cudnnBackendFinalize(engine);

  cudnnBackendDescriptor_t cfg;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &cfg);
  cudnnBackendSetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine);
  cudnnBackendFinalize(cfg);

  cudnnBackendDescriptor_t plan;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan);
  cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
  cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfg);
  cudnnBackendFinalize(plan);

  // data + variant pack
  float x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float w[4] = {1, 0, 0, 1};
  float y[4] = {0, 0, 0, 0};
  int64_t uids[3] = {1, 2, 3};
  void *ptrs[3] = {x, w, y};
  cudnnBackendDescriptor_t vp;
  cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vp);
  cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
  cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
  cudnnBackendFinalize(vp);

  CK(cudnnBackendExecute(handle, plan, vp));

  float ref[4] = {6, 8, 12, 14};
  int ok = 1;
  for (int i = 0; i < 4; i++) {
    if (y[i] != ref[i]) ok = 0;
  }
  printf("Y = [%.1f %.1f %.1f %.1f]  expected [6 8 12 14]\n", y[0], y[1], y[2], y[3]);
  printf("%s\n", ok ? "PASS — real cuDNN backend conv via oneDNN" : "MISMATCH");
  return ok ? 0 : 1;
}
