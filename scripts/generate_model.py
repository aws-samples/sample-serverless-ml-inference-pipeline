#!/usr/bin/env python3
"""Generate small, self-contained sample ONNX models for the pattern.

Each model is a linear classifier (Gemm + Softmax) that maps a feature vector of
length ``--features`` to ``--classes`` class probabilities. Three models are
produced with different random seeds so the ensemble members disagree in a
realistic way:

    model.onnx     -> single-model /predict endpoint
    model_a.onnx   -> ensemble branch A
    model_b.onnx   -> ensemble branch B

Usage:
    python scripts/generate_model.py --out-dir models_local
    aws s3 cp models_local/ s3://<ModelBucketName>/models/ --recursive
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_model(path: str, seed: int, n_features: int, n_classes: int) -> None:
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(n_features, n_classes)).astype(np.float32)
    bias = rng.normal(size=(n_classes,)).astype(np.float32)

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, n_features]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, n_classes]
    )

    weight_init = numpy_helper.from_array(weights, name="W")
    bias_init = numpy_helper.from_array(bias, name="B")

    gemm = helper.make_node("Gemm", inputs=["input", "W", "B"], outputs=["logits"])
    softmax = helper.make_node("Softmax", inputs=["logits"], outputs=["output"], axis=1)

    graph = helper.make_graph(
        nodes=[gemm, softmax],
        name="linear_classifier",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_init, bias_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="apg-sample"
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Wrote {path}  (features={n_features}, classes={n_classes}, seed={seed})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="models_local")
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--classes", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    build_model(os.path.join(args.out_dir, "model.onnx"), 0, args.features, args.classes)
    build_model(os.path.join(args.out_dir, "model_a.onnx"), 1, args.features, args.classes)
    build_model(os.path.join(args.out_dir, "model_b.onnx"), 2, args.features, args.classes)

    print(
        "\nNext: upload the models to the bucket's models/ prefix, e.g.\n"
        f"  aws s3 cp {args.out_dir}/ s3://<ModelBucketName>/models/ --recursive"
    )


if __name__ == "__main__":
    main()
