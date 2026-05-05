// RUN: mlir-opt %s -allow-unregistered-dialect \
// RUN:   -pass-pipeline='builtin.module(test.symbol_scope_isolated(test-one-shot-module-bufferize))' \
// RUN:   -split-input-file | FileCheck %s

// Verify that `tensorEncodingToMemRefLayoutFn` is consulted by the built-in
// tensor-to-memref conversion paths driven by the tensor value itself:
//   * `BuiltinTensorExternalModel::getBufferType` (used e.g. when SCF ops
//     compute the buffer type of a yielded tensor value),
//   * `bufferization.alloc_tensor`.
// The test pass installs a hook that maps `#test.tensor_encoding<...>` onto
// `#test.memref_layout<...>`.

#enc = #test.tensor_encoding<"hello">

// `bufferization.alloc_tensor` with an encoded result type must honor the
// encoding-derived layout attribute when allocating the backing memref.

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @alloc_tensor_with_encoding
  //  CHECK-SAME:  (%[[SZ:.*]]: index) -> memref<?xf32, #test.memref_layout<"hello">>
  func.func @alloc_tensor_with_encoding(%sz: index) -> tensor<?xf32, #enc> {
    // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[SZ]]) {{.*}} : memref<?xf32, #test.memref_layout<"hello">>
    %0 = bufferization.alloc_tensor(%sz) : tensor<?xf32, #enc>
    // CHECK: return %[[ALLOC]] : memref<?xf32, #test.memref_layout<"hello">>
    return %0 : tensor<?xf32, #enc>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

// An `scf.if` whose result is an encoded tensor must, on the `getBufferType`
// path, query the encoding hook when the then-branch yields a freshly
// allocated tensor with that encoding.

#enc = #test.tensor_encoding<"hello">

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @scf_if_alloc_with_encoding
  //  CHECK-SAME:  %[[COND:.*]]: i1,
  //  CHECK-SAME:  %[[ARG:.*]]: memref<?xf32, #test.memref_layout<"hello">>,
  //  CHECK-SAME:  %[[SZ:.*]]: index
  //  CHECK-SAME:  ) -> memref<?xf32, #test.memref_layout<"hello">>
  func.func @scf_if_alloc_with_encoding(
      %cond: i1, %arg: tensor<?xf32, #enc>, %sz: index) -> tensor<?xf32, #enc> {
    // CHECK: %[[R:.*]] = scf.if %[[COND]] -> (memref<?xf32, #test.memref_layout<"hello">>)
    %r = scf.if %cond -> tensor<?xf32, #enc> {
      // CHECK: scf.yield %[[ARG]]
      scf.yield %arg : tensor<?xf32, #enc>
    } else {
      // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[SZ]]) {{.*}} : memref<?xf32, #test.memref_layout<"hello">>
      %alloc = bufferization.alloc_tensor(%sz) : tensor<?xf32, #enc>
      // CHECK: scf.yield %[[ALLOC]]
      scf.yield %alloc : tensor<?xf32, #enc>
    }
    // CHECK: return %[[R]] : memref<?xf32, #test.memref_layout<"hello">>
    return %r : tensor<?xf32, #enc>
  }
  "test.finish" () : () -> ()
}) : () -> ()
