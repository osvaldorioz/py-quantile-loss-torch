#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>



namespace py = pybind11;

// Función para calcular la pérdida cuantil
py::array_t<double> quantile_loss(const py::array_t<double>& y_true, const py::array_t<double>& y_pred, double quantile) {
    auto buf_true = y_true.request();
    auto buf_pred = y_pred.request();

    if (buf_true.size != buf_pred.size) {
        throw std::runtime_error("Las entradas deben tener el mismo tamaño.");
    }

    auto ptr_true = static_cast<double*>(buf_true.ptr);
    auto ptr_pred = static_cast<double*>(buf_pred.ptr);

    size_t n = buf_true.size;
    py::array_t<double> result(n);
    auto ptr_result = static_cast<double*>(result.request().ptr);

    for (size_t i = 0; i < n; ++i) {
        double diff = ptr_true[i] - ptr_pred[i];
        if (diff > 0) {
            ptr_result[i] = quantile * diff;
        } else {
            ptr_result[i] = (quantile - 1) * diff;
        }
    }

    return result;
}

PYBIND11_MODULE(quantile_cpp, m) {
    m.def("quantile_loss", &quantile_loss, "Cálculo de pérdida cuantil",
          py::arg("y_true"), py::arg("y_pred"), py::arg("quantile"));
}
