// RUN: mlir-opt %s -allow-unregistered-dialect \
// RUN:   -pass-pipeline='builtin.module(test.symbol_scope_isolated(test-one-shot-module-bufferize))' \
// RUN:   -split-input-file | FileCheck %s

// Verify `reconcileBufferTypeMismatchFn` on the three SCF merge call sites:
//   * scf.if branches (`IfOpInterface::getBufferType`),
//   * scf.index_switch default/case (`IndexSwitchOpInterface::getBufferType`),
//   * scf.for iter_arg (`computeLoopRegionIterArgBufferType`).
//
// The mismatch is constructed by feeding each branch/iter_arg a
// `bufferization.to_tensor` produced from function arguments that carry two
// *different* custom memref layouts (`#test.memref_layout<"hello">` vs
// `#test.memref_layout<"world">`). Each branch yields a plain
// `tensor<?xf32>` whose buffer type is therefore the source memref type. The
// merge point sees two competing custom layouts and the hook has to pick
// one.
//
// The test pass installs a hook that keeps the side with the lexicographically
// smaller dummy string, so "hello" always wins -- independent of which side
// (lhs/rhs) the framework happens to hand to the hook. Without the hook, the
// framework would promote to fully-dynamic strided and the return type would
// change to `memref<?xf32, strided<[?], offset: ?>>`.
//
// Note: per the RFC discussion
// (https://discourse.llvm.org/t/rfc-generalizing-buffer-type-resolution-in-one-shot-bufferize/90393),
// the long-term default for `reconcileBufferTypeMismatchFn` is expected to
// be alloc+copy, with the hook also able to return failure or emit a cast.
// This test only exercises the "pick one side" branch of the policy, which
// is the mode downstream dialects with custom layout attributes (e.g. the
// `order` attribute in the Intel NPU compiler example) need in order to
// preserve their layout through SCF merge points. It also deliberately keeps
// the memref memory space empty so it does not conflate tensor encoding
// with memory space (those are orthogonal policies in the RFC).

#hello = #test.memref_layout<"hello">
#world = #test.memref_layout<"world">

// scf.if: "hello" in the then-branch, "world" in the else-branch.

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @reconcile_scf_if_hello_then(
  //  CHECK-SAME:     %[[COND:.*]]: i1,
  //  CHECK-SAME:     %[[MHELLO:.*]]: memref<?xf32, #test.memref_layout<"hello">>,
  //  CHECK-SAME:     %[[MWORLD:.*]]: memref<?xf32, #test.memref_layout<"world">>)
  //  CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[R:.*]] = scf.if %[[COND]] -> (memref<?xf32, #test.memref_layout<"hello">>)
  //       CHECK:   return %[[R]] : memref<?xf32, #test.memref_layout<"hello">>
  func.func @reconcile_scf_if_hello_then(
      %cond: i1,
      %mHello: memref<?xf32, #hello>,
      %mWorld: memref<?xf32, #world>) -> tensor<?xf32> {
    %a = bufferization.to_tensor %mHello restrict writable
        : memref<?xf32, #hello> to tensor<?xf32>
    %b = bufferization.to_tensor %mWorld restrict writable
        : memref<?xf32, #world> to tensor<?xf32>
    %r = scf.if %cond -> tensor<?xf32> {
      scf.yield %a : tensor<?xf32>
    } else {
      scf.yield %b : tensor<?xf32>
    }
    return %r : tensor<?xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

// Encodings swapped across branches: "hello" in the else-branch, "world" in
// the then-branch. If the hook were just "always keep lhs", the result would
// flip to "world"; instead "hello" must still win, because the policy is
// value-based.

#hello = #test.memref_layout<"hello">
#world = #test.memref_layout<"world">

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @reconcile_scf_if_hello_else(
  //  CHECK-SAME:     %[[COND:.*]]: i1,
  //  CHECK-SAME:     %[[MHELLO:.*]]: memref<?xf32, #test.memref_layout<"hello">>,
  //  CHECK-SAME:     %[[MWORLD:.*]]: memref<?xf32, #test.memref_layout<"world">>)
  //  CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[R:.*]] = scf.if %[[COND]] -> (memref<?xf32, #test.memref_layout<"hello">>)
  //       CHECK:   return %[[R]] : memref<?xf32, #test.memref_layout<"hello">>
  func.func @reconcile_scf_if_hello_else(
      %cond: i1,
      %mHello: memref<?xf32, #hello>,
      %mWorld: memref<?xf32, #world>) -> tensor<?xf32> {
    %a = bufferization.to_tensor %mHello restrict writable
        : memref<?xf32, #hello> to tensor<?xf32>
    %b = bufferization.to_tensor %mWorld restrict writable
        : memref<?xf32, #world> to tensor<?xf32>
    %r = scf.if %cond -> tensor<?xf32> {
      scf.yield %b : tensor<?xf32>
    } else {
      scf.yield %a : tensor<?xf32>
    }
    return %r : tensor<?xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

// scf.index_switch: case 0 yields "world", default yields "hello". "hello"
// wins regardless of which region is lhs in the merge.

#hello = #test.memref_layout<"hello">
#world = #test.memref_layout<"world">

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @reconcile_scf_index_switch(
  //  CHECK-SAME:     %[[IDX:.*]]: index,
  //  CHECK-SAME:     %[[MHELLO:.*]]: memref<?xf32, #test.memref_layout<"hello">>,
  //  CHECK-SAME:     %[[MWORLD:.*]]: memref<?xf32, #test.memref_layout<"world">>)
  //  CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[R:.*]] = scf.index_switch %[[IDX]] -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   return %[[R]] : memref<?xf32, #test.memref_layout<"hello">>
  func.func @reconcile_scf_index_switch(
      %idx: index,
      %mHello: memref<?xf32, #hello>,
      %mWorld: memref<?xf32, #world>) -> tensor<?xf32> {
    %a = bufferization.to_tensor %mHello restrict writable
        : memref<?xf32, #hello> to tensor<?xf32>
    %b = bufferization.to_tensor %mWorld restrict writable
        : memref<?xf32, #world> to tensor<?xf32>
    %r = scf.index_switch %idx -> tensor<?xf32>
    case 0 {
      scf.yield %b : tensor<?xf32>
    }
    default {
      scf.yield %a : tensor<?xf32>
    }
    return %r : tensor<?xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()

// -----

// scf.for iter_args: exercises `computeLoopRegionIterArgBufferType`. The
// init is seeded from the "world" memref and the loop body materializes a
// fresh tensor from the "hello" memref via `bufferization.to_tensor ...
// restrict writable`, then yields it. The init and yielded buffer types
// therefore carry two different custom layouts, the hook fires on the
// iter_arg merge and the lex-min policy ("hello" wins) promotes the
// iter_arg, the loop result, and the function result to
// `memref<?xf32, #hello>`; the init is cast accordingly.

#hello = #test.memref_layout<"hello">
#world = #test.memref_layout<"world">

"test.symbol_scope_isolated"() ({
  // CHECK-LABEL: func @reconcile_scf_for(
  //  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index,
  //  CHECK-SAME:     %[[MHELLO:.*]]: memref<?xf32, #test.memref_layout<"hello">>,
  //  CHECK-SAME:     %[[MWORLD:.*]]: memref<?xf32, #test.memref_layout<"world">>)
  //  CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[CAST:.*]] = memref.cast %[[MWORLD]]
  //  CHECK-SAME:     : memref<?xf32, #test.memref_layout<"world">> to memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[CAST]]) -> (memref<?xf32, #test.memref_layout<"hello">>)
  //       CHECK:     scf.yield %[[MHELLO]] : memref<?xf32, #test.memref_layout<"hello">>
  //       CHECK:   return %[[R]] : memref<?xf32, #test.memref_layout<"hello">>
  func.func @reconcile_scf_for(
      %lb: index, %ub: index, %step: index,
      %mHello: memref<?xf32, #hello>,
      %mWorld: memref<?xf32, #world>) -> tensor<?xf32> {
    // Init seeded from the "world" memref -> initArg bufferizes to
    // memref<?xf32, #world>.
    %init = bufferization.to_tensor %mWorld restrict writable
        : memref<?xf32, #world> to tensor<?xf32>
    %r = scf.for %i = %lb to %ub step %step
        iter_args(%it = %init) -> tensor<?xf32> {
      // Body materializes a fresh tensor from the "hello" memref and yields
      // it -> yieldedValue bufferizes to memref<?xf32, #hello>.
      %y = bufferization.to_tensor %mHello restrict writable
          : memref<?xf32, #hello> to tensor<?xf32>
      scf.yield %y : tensor<?xf32>
    }
    return %r : tensor<?xf32>
  }
  "test.finish" () : () -> ()
}) : () -> ()

