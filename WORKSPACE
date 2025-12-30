workspace(name = "quantum_drift_control")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Load rules_python for hermetic Python builds
http_archive(
    name = "rules_python",
    sha256 = "c68bdc4fbec25de5b5493b8819cfc877c4ea299c0dcb15c244c5a00208cde311",
    strip_prefix = "rules_python-0.31.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.31.0/rules_python-0.31.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")

# Point to your requirements.txt
pip_parse(
    name = "pypi",
    requirements_lock = "//:requirements.txt",
)

load("@pypi//:requirements.bzl", "install_deps")
install_deps()