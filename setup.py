from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "agentic_or._cxx_engine",
        [
            "src/bindings.cpp",
            "src/alns_operators.cpp",
        ],
        extra_compile_args=["-O3", "-std=c++20", "-march=native", "-Wall"],
    ),
]

setup(
    name="desktop-agent-or",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

