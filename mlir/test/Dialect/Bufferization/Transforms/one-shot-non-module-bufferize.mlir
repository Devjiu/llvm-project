// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline='builtin.module(test.symbol_scope_isolated(test-one-shot-module-bufferize))' -split-input-file | FileCheck %s

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @inner_func(
  //  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
  func.func @inner_func(%t: tensor<?xf32>) -> (tensor<?xf32>, f32) {
    // CHECK-NOT: copy
    %f = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: memref.store %{{.*}}, %[[arg0]]
    %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
    // CHECK: %[[load:.*]] = memref.load %[[arg0]]
    %1 = tensor.extract %0[%c1] : tensor<?xf32>
    // CHECK: return %[[arg0]], %[[load]] : memref<?xf32{{.*}}>, f32
    return %0, %1 : tensor<?xf32>, f32
  }

  // CHECK-LABEL: func @call_func_with_non_tensor_return(
  //  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
  func.func @call_func_with_non_tensor_return(
      %t0: tensor<?xf32> {bufferization.writable = true}) -> (f32, tensor<?xf32>) {
    // CHECK-NOT: alloc
    // CHECK-NOT: copy
    // CHECK: %[[call:.*]]:2 = call @inner_func(%[[arg0]])
    %0, %1 = call @inner_func(%t0) : (tensor<?xf32>) -> (tensor<?xf32>, f32)
    // CHECK: return %[[call]]#1, %[[call]]#0 : f32, memref<?xf32{{.*}}>
    return %1, %0 : f32, tensor<?xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

#enc1 = #test.tensor_encoding<"hello">
#enc2 = #test.tensor_encoding<"not hello">

"test.symbol_scope_isolated"() ({
  // CHECK: func @inner_func(
  // CHECK-SAME:  %[[arg0:.*]]: memref<?xf32, #test.memref_layout<"hello">>)
  // CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"hello">>
  func.func @inner_func(%t: tensor<?xf32, #enc1>)
      -> tensor<?xf32, #enc1> {
    // CHECK: return %[[arg0]]
    return %t : tensor<?xf32, #enc1>
  }

  // CHECK: func @outer_func(
  // CHECK-SAME:  %[[arg0:.*]]: memref<?xf32, #test.memref_layout<"hello">>)
  // CHECK-SAME:  -> (memref<?xf32, #test.memref_layout<"hello">>,
  // CHECK-SAME:      memref<?xf32, #test.memref_layout<"not hello">>)
  func.func @outer_func(%t0: tensor<?xf32, #enc1>)
      -> (tensor<?xf32, #enc1>, tensor<?xf32, #enc2>) {
    // CHECK: %[[call:.*]] = call @inner_func(%[[arg0]])
    %0 = call @inner_func(%t0)
      : (tensor<?xf32, #enc1>) -> (tensor<?xf32, #enc1>)

    // CHECK: %[[local:.*]] = "test.create_memref_op"() : ()
    // CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"not hello">>
    %local = "test.create_tensor_op"() : () -> tensor<?xf32, #enc2>
    // CHECK: %[[dummy:.*]] = "test.dummy_memref_op"(%[[local]])
    %1 = "test.dummy_tensor_op"(%local) : (tensor<?xf32, #enc2>)
      -> tensor<?xf32, #enc2>

    // CHECK: return %[[call]], %[[dummy]]
    return %0, %1 : tensor<?xf32, #enc1>, tensor<?xf32, #enc2>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

#enc_stride_2d = #test.tensor_encoding<strides = [17, 3], offset = 5>

"test.symbol_scope_isolated"() ({
  func.func private @builtin_encoding_identity_2d(
      %arg: tensor<4x4xf64, #enc_stride_2d>)
      -> tensor<4x4xf64, #enc_stride_2d> {
    return %arg : tensor<4x4xf64, #enc_stride_2d>
  }

  // CHECK-LABEL: func.func @builtin_encoding_scf_for_inplace_with_call(
  // CHECK-SAME: %[[arg:.+]]: memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK-SAME: %[[lb:.+]]: index, %[[ub:.+]]: index, %[[step:.+]]: index
  // CHECK-SAME: ) -> memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: %[[loop:.+]] = scf.for %{{.*}} = %[[lb]] to %[[ub]] step %[[step]] iter_args(%[[iter:.+]] = %[[arg]]) -> (memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>) {
  // CHECK: %[[call:.+]] = func.call @builtin_encoding_identity_2d(%[[iter]]) : (memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>) -> memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: scf.yield %[[call]] : memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: return %[[loop]] : memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  func.func @builtin_encoding_scf_for_inplace_with_call(
      %arg: tensor<4x4xf64, #enc_stride_2d>,
      %lb: index, %ub: index, %step: index)
      -> tensor<4x4xf64, #enc_stride_2d> {
    %loop = scf.for %i = %lb to %ub step %step
        iter_args(%iter = %arg) -> (tensor<4x4xf64, #enc_stride_2d>) {
      %call = func.call @builtin_encoding_identity_2d(%iter)
        : (tensor<4x4xf64, #enc_stride_2d>) -> tensor<4x4xf64, #enc_stride_2d>
      scf.yield %call : tensor<4x4xf64, #enc_stride_2d>
    }
    return %loop : tensor<4x4xf64, #enc_stride_2d>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

#enc_stride_2d = #test.tensor_encoding<strides = [17, 3], offset = 5>

"test.symbol_scope_isolated"() ({
  func.func private @builtin_encoding_identity_2d(
      %arg: tensor<4x4xf64, #enc_stride_2d>)
      -> tensor<4x4xf64, #enc_stride_2d> {
    return %arg : tensor<4x4xf64, #enc_stride_2d>
  }

  // CHECK-LABEL: func.func @builtin_encoding_scf_if_inplace_with_call(
  // CHECK-SAME: %[[arg:.+]]: memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK-SAME: %[[cond:.+]]: i1
  // CHECK-SAME: ) -> memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: %[[res:.+]] = scf.if %[[cond]] -> (memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>) {
  // CHECK: %[[call:.+]] = func.call @builtin_encoding_identity_2d(%[[arg]]) : (memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>) -> memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: scf.yield %[[call]] : memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: } else {
  // CHECK: scf.yield %[[arg]] : memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: }
  // CHECK: return %[[res]] : memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  func.func @builtin_encoding_scf_if_inplace_with_call(
      %arg: tensor<4x4xf64, #enc_stride_2d>,
      %cond: i1)
      -> tensor<4x4xf64, #enc_stride_2d> {
    %res = scf.if %cond -> (tensor<4x4xf64, #enc_stride_2d>) {
      %call = func.call @builtin_encoding_identity_2d(%arg)
        : (tensor<4x4xf64, #enc_stride_2d>) -> tensor<4x4xf64, #enc_stride_2d>
      scf.yield %call : tensor<4x4xf64, #enc_stride_2d>
    } else {
      scf.yield %arg : tensor<4x4xf64, #enc_stride_2d>
    }
    return %res : tensor<4x4xf64, #enc_stride_2d>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

#enc_stride_2d = #test.tensor_encoding<strides = [17, 3], offset = 5>

"test.symbol_scope_isolated"() ({
  func.func private @builtin_encoding_identity_2d(
      %arg: tensor<4x4xf64, #enc_stride_2d>)
      -> tensor<4x4xf64, #enc_stride_2d> {
    return %arg : tensor<4x4xf64, #enc_stride_2d>
  }

  // CHECK-LABEL: func.func @builtin_encoding_scf_while_inplace_with_call(
  // CHECK-SAME: %[[arg:.+]]: memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK-SAME: %[[cond:.+]]: i1
  // CHECK-SAME: ) -> memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: %[[loop:.+]] = scf.while (%[[iter:.+]] = %[[arg]]) : (memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>) -> memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>> {
  // CHECK: scf.condition(%[[cond]]) %[[iter]] : memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: } do {
  // CHECK: ^bb0(%[[current:.+]]: memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>):
  // CHECK: %[[call:.+]] = func.call @builtin_encoding_identity_2d(%[[current]]) : (memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>) -> memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: scf.yield %[[call]] : memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  // CHECK: }
  // CHECK: return %[[loop]] : memref<4x4xf64, #test.memref_layout<strides = [17, 3], offset = 5>>
  func.func @builtin_encoding_scf_while_inplace_with_call(
      %arg: tensor<4x4xf64, #enc_stride_2d>,
      %cond: i1)
      -> tensor<4x4xf64, #enc_stride_2d> {
    %loop = scf.while (%iter = %arg)
        : (tensor<4x4xf64, #enc_stride_2d>) -> tensor<4x4xf64, #enc_stride_2d> {
      scf.condition(%cond) %iter : tensor<4x4xf64, #enc_stride_2d>
    } do {
    ^bb0(%current: tensor<4x4xf64, #enc_stride_2d>):
      %call = func.call @builtin_encoding_identity_2d(%current)
        : (tensor<4x4xf64, #enc_stride_2d>) -> tensor<4x4xf64, #enc_stride_2d>
      scf.yield %call : tensor<4x4xf64, #enc_stride_2d>
    }
    return %loop : tensor<4x4xf64, #enc_stride_2d>
  }
  "test.finish" () : () -> ()
}) : () -> ()
