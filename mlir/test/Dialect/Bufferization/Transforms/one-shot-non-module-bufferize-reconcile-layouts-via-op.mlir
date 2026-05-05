// RUN: mlir-opt %s -allow-unregistered-dialect \
// RUN:   -pass-pipeline='builtin.module(test.symbol_scope_isolated(test-one-shot-module-bufferize))' \
// RUN:   -split-input-file | FileCheck %s

// Companion to one-shot-non-module-bufferize-reconcile-layouts.mlir.
//
// That file constructs the SCF buffer-type mismatch by pre-bufferizing
// function arguments via `bufferization.to_tensor ... restrict writable`.
// This file constructs the same mismatch at a higher level: a regular
// tensor-typed op (`test.tensor_with_layout`) declares its bufferized memref
// layout via a string attribute through `BufferizableOpInterface::getBufferType`.
// Because the *tensor* result type stays plain (no encoding), the SCF
// verifier accepts identical iter_arg/init/yield tensor types, while the
// *bufferized* memref types differ in layout -- which is what
// `tryReconcileBufferType` is supposed to handle.
//
// The lex-min hook installed by the test pass keeps "hello" over "world",
// so all three SCF call sites must surface
// `memref<?xf32, #test.memref_layout<"hello">>` at the merge point.

// scf.if: then = "world", else = "hello".

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @reconcile_op_scf_if(
  //  CHECK-SAME:     %[[COND:.*]]: i1)
  //  CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[R:.*]] = scf.if %[[COND]] -> (memref<?xf32, #test.memref_layout<"hello">>)
  //       CHECK:   return %[[R]] : memref<?xf32, #test.memref_layout<"hello">>
  func.func @reconcile_op_scf_if(%cond: i1) -> tensor<?xf32> {
    %r = scf.if %cond -> tensor<?xf32> {
      %w = "test.tensor_with_layout"() {layout = "world"} : () -> tensor<?xf32>
      scf.yield %w : tensor<?xf32>
    } else {
      %h = "test.tensor_with_layout"() {layout = "hello"} : () -> tensor<?xf32>
      scf.yield %h : tensor<?xf32>
    }
    return %r : tensor<?xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

// scf.index_switch: case 0 = "world", default = "hello".

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @reconcile_op_scf_index_switch(
  //  CHECK-SAME:     %[[IDX:.*]]: index)
  //  CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[R:.*]] = scf.index_switch %[[IDX]] -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   return %[[R]] : memref<?xf32, #test.memref_layout<"hello">>
  func.func @reconcile_op_scf_index_switch(%idx: index) -> tensor<?xf32> {
    %r = scf.index_switch %idx -> tensor<?xf32>
    case 0 {
      %w = "test.tensor_with_layout"() {layout = "world"} : () -> tensor<?xf32>
      scf.yield %w : tensor<?xf32>
    }
    default {
      %h = "test.tensor_with_layout"() {layout = "hello"} : () -> tensor<?xf32>
      scf.yield %h : tensor<?xf32>
    }
    return %r : tensor<?xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

// scf.for iter_args: init = "world", yielded body value = "hello". The
// reconcile hook promotes the iter_arg/result to "hello" and the framework
// must cast the init memref accordingly.

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @reconcile_op_scf_for(
  //  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
  //  CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[INIT:.*]] = "test.create_memref_op"() : () -> memref<?xf32, #test.memref_layout<"world">>
  //       CHECK:   %[[CAST:.*]] = memref.cast %[[INIT]]
  //  CHECK-SAME:     : memref<?xf32, #test.memref_layout<"world">> to memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[CAST]]) -> (memref<?xf32, #test.memref_layout<"hello">>)
  //       CHECK:     %[[H:.*]] = "test.create_memref_op"() : () -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:     scf.yield %[[H]] : memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   return %[[R]] : memref<?xf32, #test.memref_layout<"hello">>
  func.func @reconcile_op_scf_for(
      %lb: index, %ub: index, %step: index) -> tensor<?xf32> {
    %init = "test.tensor_with_layout"() {layout = "world"} : () -> tensor<?xf32>
    %r = scf.for %i = %lb to %ub step %step
        iter_args(%it = %init) -> tensor<?xf32> {
      %y = "test.tensor_with_layout"() {layout = "hello"} : () -> tensor<?xf32>
      scf.yield %y : tensor<?xf32>
    }
    return %r : tensor<?xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

// scf.for + tensor.extract_slice/insert_slice. This mirrors the realistic
// downstream pattern (cf. the NPU `VPU.Convert` tiling example): the loop's
// init buffer carries one hardware layout ("world"), while the per-iteration
// destination buffer is allocated by a layout-aware producer with a
// different layout ("hello"). The slice/insert pair feeds the body and the
// yielded value carries the "hello" layout, so the iter_arg merge sees a
// real mismatch. After reconciliation the iter_arg, loop result and func
// result are all promoted to "hello"; the init is `memref.cast`-ed and the
// body lowers to a `memref.subview` + `memref.copy` pair via the standard
// `tensor.insert_slice` bufferization.

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @reconcile_op_scf_for_extract_insert_slice(
  //  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
  //  CHECK-SAME:  -> memref<128xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[INIT:.*]] = "test.create_memref_op"() : () -> memref<128xf32, #test.memref_layout<"world">>
  //       CHECK:   %[[CAST:.*]] = memref.cast %[[INIT]]
  //  CHECK-SAME:     : memref<128xf32, #test.memref_layout<"world">> to memref<128xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[R:.*]] = scf.for {{.*}} iter_args(%[[IT:.*]] = %[[CAST]]) -> (memref<128xf32, #test.memref_layout<"hello">>)
  //       CHECK:     %[[SRC:.*]] = memref.subview %[[IT]]
  //       CHECK:     %[[DST_BUF:.*]] = "test.create_memref_op"() : () -> memref<128xf32, #test.memref_layout<"hello">>
  //       CHECK:     %[[DST:.*]] = memref.subview %[[DST_BUF]]
  //       CHECK:     memref.copy %[[SRC]], %[[DST]]
  //       CHECK:     scf.yield %[[DST_BUF]] : memref<128xf32, #test.memref_layout<"hello">>
  //       CHECK:   return %[[R]] : memref<128xf32, #test.memref_layout<"hello">>
  func.func @reconcile_op_scf_for_extract_insert_slice(
      %lb: index, %ub: index, %step: index) -> tensor<128xf32> {
    %c0 = arith.constant 0 : index
    %init = "test.tensor_with_layout"() {layout = "world"} : () -> tensor<128xf32>
    %r = scf.for %i = %lb to %ub step %step
        iter_args(%it = %init) -> tensor<128xf32> {
      %slice = tensor.extract_slice %it[%c0] [32] [1]
          : tensor<128xf32> to tensor<32xf32>
      %dest = "test.tensor_with_layout"() {layout = "hello"} : () -> tensor<128xf32>
      %y = tensor.insert_slice %slice into %dest[%c0] [32] [1]
          : tensor<32xf32> into tensor<128xf32>
      scf.yield %y : tensor<128xf32>
    }
    return %r : tensor<128xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()
