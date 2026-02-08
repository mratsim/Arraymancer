// Copyright 2017 the Arraymancer contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Element-wise Operations (float32 only - Metal doesn't support float64)
// ============================================================================

kernel void elementwise_add_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        C[gid] = A[gid] + B[gid];
    }
}

kernel void elementwise_sub_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        C[gid] = A[gid] - B[gid];
    }
}

kernel void elementwise_mul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        C[gid] = A[gid] * B[gid];
    }
}

kernel void elementwise_div_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        C[gid] = A[gid] / B[gid];
    }
}

// ============================================================================
// Scalar Operations
// ============================================================================

kernel void scalar_mul_f32(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        C[gid] = A[gid] * scalar;
    }
}

kernel void scalar_add_f32(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        C[gid] = A[gid] + scalar;
    }
}

// ============================================================================
// Simple GEMM (naive implementation for reference)
// Column-major layout (Fortran style)
// ============================================================================

kernel void gemm_naive_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;  // 0 to M-1
    uint col = gid.x;  // 0 to N-1

    if (row < (uint)M && col < (uint)N) {
        float sum = 0.0;
        // Column-major indexing:
        // A[i,k] is at A[k*M + i]
        // B[k,j] is at B[j*K + k]
        // C[i,j] is at C[j*M + i]
        for (int k = 0; k < K; k++) {
            sum += A[k * M + row] * B[col * K + k];
        }
        C[col * M + row] = alpha * sum + beta * C[col * M + row];
    }
}

// ============================================================================
// Transpose GEMM variants
// Column-major layout (Fortran style)
// ============================================================================

kernel void gemm_a_transpose_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row < (uint)M && col < (uint)N) {
        float sum = 0.0;
        // A^T[i,k] = A[k,i], in column-major: A[k,i] is at A[i*K + k]
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[col * M + row] = alpha * sum + beta * C[col * M + row];
    }
}

kernel void gemm_b_transpose_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row < (uint)M && col < (uint)N) {
        float sum = 0.0;
        // B^T[k,j] = B[j,k], in column-major: B[j,k] is at B[k*N + j]
        for (int k = 0; k < K; k++) {
            sum += A[k * M + row] * B[k * N + col];
        }
        C[col * M + row] = alpha * sum + beta * C[col * M + row];
    }
}

// ============================================================================
// Batched GEMM (for future use with 3D+ tensors)
// ============================================================================

kernel void batched_gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& batchSize [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant float& alpha [[buffer(7)]],
    constant float& beta [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (batch < (uint)batchSize && row < (uint)M && col < (uint)N) {
        int batchOffsetA = batch * M * K;
        int batchOffsetB = batch * K * N;
        int batchOffsetC = batch * M * N;

        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[batchOffsetA + row * K + k] * B[batchOffsetB + k * N + col];
        }
        C[batchOffsetC + row * N + col] = alpha * sum + beta * C[batchOffsetC + row * N + col];
    }
}

// ============================================================================
// Vector Operations (BLAS Level 1)
// ============================================================================

kernel void vector_dot_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]])
{
    threadgroup float localSum[256];

    float sum = 0.0;
    for (uint i = gid; i < (uint)size; i += groupSize) {
        sum += A[i] * B[i];
    }

    localSum[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = groupSize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            localSum[lid] += localSum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        result[gid / groupSize] = localSum[0];
    }
}

kernel void vector_scale_f32(
    device float* X [[buffer(0)]],
    constant float& alpha [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        X[gid] *= alpha;
    }
}

kernel void vector_axpy_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        Y[gid] += alpha * X[gid];
    }
}

// ============================================================================
// Activation Functions (for ML workloads)
// ============================================================================

kernel void relu_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        Y[gid] = max(0.0f, X[gid]);
    }
}

kernel void sigmoid_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        Y[gid] = 1.0f / (1.0f + exp(-X[gid]));
    }
}

kernel void tanh_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < (uint)size) {
        Y[gid] = tanh(X[gid]);
    }
}

// ============================================================================
// Contiguous Copy Kernel
// Copies data from a strided tensor to a contiguous buffer
// ============================================================================

kernel void contiguous_copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int* src_shape [[buffer(2)]],
    constant int* src_strides [[buffer(3)]],
    constant int& rank [[buffer(4)]],
    constant int& total_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)total_size) return;
    
    // Convert flat index to multi-dimensional indices
    int idx = (int)gid;
    int src_offset = 0;
    
    for (int dim = rank - 1; dim >= 0; dim--) {
        int coord = idx % src_shape[dim];
        src_offset += coord * src_strides[dim];
        idx /= src_shape[dim];
    }
    
    dst[gid] = src[src_offset];
}

// ============================================================================
// Matrix-Vector Multiplication
// Column-major layout (Fortran style)
// y = alpha * A * x + beta * y
// ============================================================================

kernel void gemv_f32(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    constant float& beta [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    // Each thread computes one element of the result vector y
    // gid is the row index (0 to M-1)
    if (gid < (uint)M) {
        float sum = 0.0;
        // Column-major indexing:
        // A[i,j] is at A[j*M + i]
        // x[j] is at x[j]
        // y[i] is at y[i]
        for (int j = 0; j < N; j++) {
            sum += A[j * M + gid] * x[j];
        }
        y[gid] = alpha * sum + beta * y[gid];
    }
}
