from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "quantile_cpp",
        ["app/quantile_loss.cpp"],
        extra_compile_args=["-std=c++20"]
    ),
]

setup(
    name="quantile_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
