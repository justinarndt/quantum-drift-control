load("@rules_python//python:defs.bzl", "py_library", "py_binary")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "stim_integration",
    srcs = ["stim_integration.py"],
    deps = [
        "//src/drift:orchestrator",
        "@pypi//absl_py",
        "@pypi//numpy",
        "@pypi//stim",
    ],
)

py_binary(
    name = "stream_generator",
    srcs = ["stream_generator.py"],
    deps = [
        ":stim_integration",
        "//src/drift:orchestrator",
        "@pypi//absl_py",
    ],
)