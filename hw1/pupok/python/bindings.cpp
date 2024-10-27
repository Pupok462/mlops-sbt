#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cosine_similarity.h"

namespace py = pybind11;

PYBIND11_MODULE(pupok_core, m) {
    m.doc() = R"doc(
    Module for computing cosine similarity between two vectors
    )doc";

    m.def("cosine_similarity", &cosine_similarity, py::arg("vec_a"), py::arg("vec_b"),
          R"doc(
              Compute the cosine similarity between two vectors.

              Parameters:
                  vec_a (list of float): The first vector.
                  vec_b (list of float): The second vector.

              Returns:
                  float: The cosine similarity between vec_a and vec_b.
          )doc");
}
