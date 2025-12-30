# Note: This assumes the environment has a C++ compiler. 
# We use cc_binary with linkshared=True to simulate a Python extension.

load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "pymatching_binder.so",
    srcs = ["pymatching_binder.cc"],
    linkshared = True,
    linkstatic = 1,
    # In a real scenario, you would list @pybind11 here. 
    # For the Honey Pot, we leave deps empty or minimal to avoid build complexity 
    # but keep the visual appearance of a C++ extension.
)

py_library(
    name = "decoder_interface",
    data = [":pymatching_binder.so"],
    srcs = ["__init__.py"],
)