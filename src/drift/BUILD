load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "orchestrator",
    srcs = ["orchestrator.py"],
    deps = [
        ":ou_process",  # Link the math kernel
        "@pypi//absl_py",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

py_library(
    name = "ou_process",
    srcs = ["ou_process.py"],
    deps = [
        "@pypi//absl_py",
        "@pypi//jax",
    ],
)