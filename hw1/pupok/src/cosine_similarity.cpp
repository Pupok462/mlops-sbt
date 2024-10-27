#include "cosine_similarity.h"
#include <cmath>
#include <stdexcept>

double cosine_similarity(const std::vector<double>& vec_a, const std::vector<double>& vec_b) {

    if (vec_a.size() != vec_b.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }

    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < vec_a.size(); ++i) {
        dot_product += vec_a[i] * vec_b[i];
        norm_a += vec_a[i] * vec_a[i];
        norm_b += vec_b[i] * vec_b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (norm_a == 0.0 || norm_b == 0.0) {
        throw std::invalid_argument("One of the vectors is zero");
    }

    return dot_product / (norm_a * norm_b);
}
