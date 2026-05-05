//===- TestOneShotModuleBufferzation.cpp - Bufferization Test -----*- c++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "TestAttributes.h" // TestTensorEncodingAttr, TestMemRefLayoutAttr
#include "TestDialect.h"

using namespace mlir;

namespace {
MemRefLayoutAttrInterface
getMemRefLayoutForTensorEncoding(RankedTensorType tensorType) {
  if (auto encoding = dyn_cast_if_present<test::TestTensorEncodingAttr>(
          tensorType.getEncoding())) {
    return cast<MemRefLayoutAttrInterface>(test::TestMemRefLayoutAttr::get(
        tensorType.getContext(), encoding.getDummy()));
  }
  return {};
}

struct TestOneShotModuleBufferizePass
    : public PassWrapper<TestOneShotModuleBufferizePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOneShotModuleBufferizePass)

  TestOneShotModuleBufferizePass() = default;
  TestOneShotModuleBufferizePass(const TestOneShotModuleBufferizePass &pass) =
      default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<test::TestDialect>();
    registry.insert<bufferization::BufferizationDialect>();
  }
  StringRef getArgument() const final {
    return "test-one-shot-module-bufferize";
  }
  StringRef getDescription() const final {
    return "Pass to test One Shot Module Bufferization";
  }

  void runOnOperation() override {

    llvm::errs() << "Running TestOneShotModuleBufferize on: "
                 << getOperation()->getName() << "\n";
    bufferization::OneShotBufferizationOptions opt;

    opt.bufferizeFunctionBoundaries = true;
    opt.allowReturnAllocsFromLoops = true;

    // Map `TestTensorEncodingAttr` on builtin tensors to the corresponding
    // `TestMemRefLayoutAttr` on the bufferized memref. Consulted by the
    // built-in tensor-to-memref conversion paths
    // (`BuiltinTensorExternalModel::getBufferType`, `alloc_tensor`).
    opt.tensorEncodingToMemRefLayoutFn =
        [](TensorType t) -> MemRefLayoutAttrInterface {
      if (auto rtt = dyn_cast<RankedTensorType>(t))
        return getMemRefLayoutForTensorEncoding(rtt);
      return {};
    };

    opt.functionArgTypeConverterFn =
        [&](bufferization::TensorLikeType tensor, Attribute memSpace,
            func::FuncOp,
            const bufferization::BufferizationOptions &options) {
          assert(isa<RankedTensorType>(tensor) && "tests only builtin tensors");
          auto tensorType = cast<RankedTensorType>(tensor);
          MemRefLayoutAttrInterface layout;
          if (options.tensorEncodingToMemRefLayoutFn)
            layout = options.tensorEncodingToMemRefLayoutFn(tensorType);
          return cast<bufferization::BufferLikeType>(
              MemRefType::get(tensorType.getShape(),
                              tensorType.getElementType(), layout, memSpace));
        };

    // Reconcile any SCF buffer-type mismatch by preferring the side whose
    // `TestMemRefLayoutAttr` dummy value is lexicographically smaller. This
    // is observably different from the framework's default fallback (promote
    // to fully-dynamic strided) and is deterministic across lhs/rhs order,
    // so tests can exercise `reconcileBufferTypeMismatchFn` on the three SCF
    // call sites (scf.if, scf.index_switch, scf.for iter_arg) with two
    // distinct test encodings and assert which one wins.
    opt.reconcileBufferTypeMismatchFn =
        [](Operation *,
           bufferization::BufferizationOptions::BufferTypeMismatchKind,
           bufferization::BufferLikeType lhs, bufferization::BufferLikeType rhs,
           const bufferization::BufferizationOptions &)
        -> FailureOr<bufferization::BufferLikeType> {
      auto testLayout = [](bufferization::BufferLikeType t) {
        auto m = dyn_cast<MemRefType>(t);
        return m ? dyn_cast<test::TestMemRefLayoutAttr>(m.getLayout())
                 : test::TestMemRefLayoutAttr();
      };
      auto lhsLayout = testLayout(lhs);
      auto rhsLayout = testLayout(rhs);
      if (lhsLayout && rhsLayout)
        return lhsLayout.getDummy().getValue() <=
                       rhsLayout.getDummy().getValue()
                   ? lhs
                   : rhs;
      if (lhsLayout)
        return lhs;
      if (rhsLayout)
        return rhs;
      return lhs;
    };

    bufferization::BufferizationState bufferizationState;

    if (failed(bufferization::runOneShotModuleBufferize(getOperation(), opt,
                                                        bufferizationState)))
      signalPassFailure();
  }
};
} // namespace

namespace mlir::test {
void registerTestOneShotModuleBufferizePass() {
  PassRegistration<TestOneShotModuleBufferizePass>();
}
} // namespace mlir::test
