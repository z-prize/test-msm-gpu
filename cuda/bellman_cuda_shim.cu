// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <iostream>
#include <nvToolsExt.h>
#include <string>
#include <sys/mman.h>

#include <ff/bls12-377.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

#include "../bellman-cuda/src/bellman-cuda.h"

#ifndef __CUDA_ARCH__

static const size_t NUM_BATCH_THREADS = 2;
static thread_pool_t batch_pool(NUM_BATCH_THREADS);

typedef pippenger_t<bucket_t, point_t, affine_t, scalar_t> pipp_t;

struct bc_affine_t {
  fp_t x;
  fp_t y;
};

struct bc_proj_t {
  fp_t x;
  fp_t y;
  fp_t z;
};

// MSM context used store persistent state
struct Context {
  bc_mem_pool mem_pool;
  bc_affine_t *d_bases;
  bool *d_bases_inf_flags;
  scalar_t *h_scalars[2];
  scalar_t *h_results[2];
  scalar_t *d_scalars[2];
  scalar_t *d_results[2];
  bc_stream streams[2];
  bc_event events[2];
};

template<class bucket_t, class affine_t, class scalar_t>
struct RustContext {
  Context *context;
};

// not pretty, especially since bc_error was itself a cast from a cudaError_t,
// but we're just trying to make our stuff work with the harness with minimal effort,
// not paint the sistine chapel
#define BC_CHK(ARG) \
{ \
  bc_error err = (ARG); \
  if (err != 0) { \
    cudaError_t code = static_cast<cudaError_t>(err); \
    return RustError(code); \
  } \
}

#define CUDA_CHK(ARG) \
{ \
  cudaError_t code = (ARG); \
  if (code != cudaSuccess) { \
    return RustError{code}; \
  } \
}

unsigned get_log_count(size_t npoints) {
  assert(npoints > 0);
  size_t tmp{npoints};
  unsigned log_count{0};
  while (tmp) {
    log_count++;
    tmp >>= 1;
  }
  return --log_count;
}

struct nvtxRangeGuard {
  nvtxRangeGuard(const char *label) { nvtxRangePush(label); }
  ~nvtxRangeGuard() { nvtxRangePop(); }
};

// Initialization function
// Allocate device storage, transfer bases
extern "C"
RustError mult_pippenger_init_bc(RustContext<bucket_t,
                                 affine_t, scalar_t> *context,
                                 const affine_t points[],
                                 const bool h_bases_inf_flags[],
                                 size_t npoints,
                                 size_t ffi_affine_sz)
{
  static_assert(sizeof(bc_proj_t) == sizeof(point_t));

  nvtxRangeGuard g{__FUNCTION__};

  // context is never deleted (leaks after timing loop in benches/msm.rs) but reference maintainer says that's ok.
  context->context = new Context{};
  auto *ctx = context->context;

  BC_CHK(msm_set_up());

  std::cout << "npoints " << npoints << std::endl;

  BC_CHK(bc_mem_pool_create(&ctx->mem_pool, 0));
  BC_CHK(bc_malloc((void**)&ctx->d_bases, sizeof(bc_affine_t) * npoints));
  BC_CHK(bc_malloc((void**)&ctx->d_bases_inf_flags, sizeof(bool) * npoints));

  for (int i = 0; i < 2; i++) {
    BC_CHK(bc_malloc_host((void**)&ctx->h_scalars[i], sizeof(scalar_t) * npoints));
    BC_CHK(bc_malloc_host((void**)&ctx->h_results[i], sizeof(bc_proj_t) * 256));
    BC_CHK(bc_malloc((void**)&ctx->d_scalars[i], sizeof(scalar_t) * npoints));
    BC_CHK(bc_malloc((void**)&ctx->d_results[i], sizeof(bc_proj_t) * 256));
    BC_CHK(bc_stream_create(&ctx->streams[i], false));
    BC_CHK(bc_event_create(&ctx->events[i], false, true));
  }

  // Copies bases from host to device.
  // Intent is to pick out limb data and skip inf flags, which are copied separately.
  CUDA_CHK(cudaMemcpy2D(ctx->d_bases, sizeof(bc_affine_t), points, ffi_affine_sz,
                        sizeof(bc_affine_t), npoints, cudaMemcpyHostToDevice));

  BC_CHK(bc_memcpy(ctx->d_bases_inf_flags, h_bases_inf_flags, sizeof(bool) * npoints));

  return RustError{cudaSuccess};
}

// Peform MSM on a batch of scalars over fixed bases
extern "C"
RustError mult_pippenger_bc(RustContext<bucket_t,
                            affine_t, scalar_t> *context,
                            point_t* out,
                            const affine_t points[],
                            size_t npoints,
                            size_t batches,
                            const scalar_t scalars[],
                            size_t ffi_affine_sz)
{
  nvtxRangeGuard g{__FUNCTION__};

  auto *ctx = context->context;

  unsigned log_count = get_log_count(npoints);
  size_t bytes = sizeof(scalar_t) * npoints;

  for (unsigned batch = 0; batch < batches; batch++) {
    nvtxRangeGuard g{std::to_string(batch).c_str()};
    unsigned i = batch & 1; // alternate buffers + streams each iteration
    if (batch > 0) {
      CUDA_CHK(cudaMemcpyAsync(ctx->h_scalars[i], scalars, bytes, cudaMemcpyHostToHost,
                               static_cast<cudaStream_t>(ctx->streams[i].handle)));
      CUDA_CHK(cudaMemcpyAsync(ctx->d_scalars[i], ctx->h_scalars[i], bytes, cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(ctx->streams[i].handle)));
    }
    msm_configuration cfg = {ctx->mem_pool,
                             ctx->streams[i],
                             ctx->d_bases,
                             batch == 0 ? (void*)scalars : (void*)ctx->d_scalars[i],
                             ctx->d_results[i],
                             log_count};
    cfg.bases_inf_flags = ctx->d_bases_inf_flags;
    BC_CHK(msm_execute_async(cfg));
    scalars += npoints; 
  }

  for (auto& stream : ctx->streams) {
    BC_CHK(bc_stream_synchronize(stream));
  }

  return RustError{cudaSuccess};
}

#endif  //  __CUDA_ARCH__
