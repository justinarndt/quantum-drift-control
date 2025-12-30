load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "drift_test",
    srcs = ["drift_test.py"],
    deps = [
        "//src/drift:ou_process",
        "@pypi//absl_py",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)