// RUN: mlir-opt %s \
// RUN:   -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries allow-unknown-ops" \
// RUN:   -split-input-file | FileCheck %s --check-prefix=DEFAULT

// RUN: mlir-opt %s \
// RUN:   -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries allow-unknown-ops reconcile-layouts-keep-lhs" \
// RUN:   -split-input-file | FileCheck %s --check-prefix=KEEP-LHS

// Verify that reconcileBufferTypeMismatchFn is invoked on every SCF op that
// has to merge two independently inferred buffer types for the same
// bufferized value: scf.if, scf.index_switch, scf.for (iter_arg).
// Each case: without the hook the framework promotes to fully-dynamic
// strided layout; with the hook installed the lhs layout is kept as-is
// (no fully-dynamic promotion).

// scf.if - then/else produce buffers with different layouts.

// DEFAULT-LABEL: func @reconcile_scf_if(
// DEFAULT:  scf.if {{.*}} -> (memref<4xf32, strided<[?], offset: ?>>)
// DEFAULT:  return {{.*}} : memref<4xf32, strided<[?], offset: ?>>

// KEEP-LHS-LABEL: func @reconcile_scf_if(
// KEEP-LHS:  scf.if {{.*}} -> (memref<4xf32, strided<[1]>>)
// KEEP-LHS:  return %{{.*}} : memref<4xf32, strided<[1]>>
func.func @reconcile_scf_if(
    %cond: i1,
    %src: memref<4xf32, strided<[1], offset: 0>>) -> tensor<4xf32> {
  %t = bufferization.to_tensor %src restrict
      : memref<4xf32, strided<[1], offset: 0>> to tensor<4xf32>
  %r = scf.if %cond -> tensor<4xf32> {
    scf.yield %t : tensor<4xf32>
  } else {
    %alloc = bufferization.alloc_tensor() : tensor<4xf32>
    scf.yield %alloc : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// -----

// scf.index_switch - default + case yield buffers with different layouts.

// DEFAULT-LABEL: func @reconcile_scf_index_switch(
// DEFAULT:  scf.index_switch {{.*}} -> memref<4xf32, strided<[?], offset: ?>>
// DEFAULT:  return {{.*}} : memref<4xf32, strided<[?], offset: ?>>

// KEEP-LHS-LABEL: func @reconcile_scf_index_switch(
// KEEP-LHS:  scf.index_switch {{.*}} -> memref<4xf32>
// KEEP-LHS:  return %{{.*}} : memref<4xf32>
func.func @reconcile_scf_index_switch(
    %idx: index,
    %src: memref<4xf32, strided<[1], offset: 0>>) -> tensor<4xf32> {
  %t = bufferization.to_tensor %src restrict
      : memref<4xf32, strided<[1], offset: 0>> to tensor<4xf32>
  %r = scf.index_switch %idx -> tensor<4xf32>
  case 0 {
    scf.yield %t : tensor<4xf32>
  }
  default {
    %alloc = bufferization.alloc_tensor() : tensor<4xf32>
    scf.yield %alloc : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// -----

// scf.for - iter_arg init carries strided layout; body yields a freshly
// allocated identity-layout tensor.

// DEFAULT-LABEL: func @reconcile_scf_for(
// DEFAULT:  scf.for {{.*}} iter_args({{.*}}) -> (memref<4xf32, strided<[?], offset: ?>>)
// DEFAULT:  return {{.*}} : memref<4xf32, strided<[?], offset: ?>>

// KEEP-LHS-LABEL: func @reconcile_scf_for(
// KEEP-LHS:  scf.for {{.*}} iter_args({{.*}}) -> (memref<4xf32, strided<[1]>>)
// KEEP-LHS:  return %{{.*}} : memref<4xf32, strided<[1]>>
func.func @reconcile_scf_for(
    %lb: index, %ub: index, %step: index,
    %src: memref<4xf32, strided<[1], offset: 0>>) -> tensor<4xf32> {
  %t = bufferization.to_tensor %src restrict writable
      : memref<4xf32, strided<[1], offset: 0>> to tensor<4xf32>
  %r = scf.for %i = %lb to %ub step %step iter_args(%it = %t) -> tensor<4xf32> {
    %alloc = bufferization.alloc_tensor() : tensor<4xf32>
    scf.yield %alloc : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}
