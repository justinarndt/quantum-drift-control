// Copyright 2025 Justin Arndt.
// Efficient graph reweighting for PyMatching 2.0.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

// Computes the log-likelihood ratio for a given error probability
// w = ln((1-p)/p)
double weight_from_probability(double p) {
    if (p < 1e-9) return 1000.0; // Max weight cap
    return std::log((1.0 - p) / p);
}

// Batch update of edge weights based on drift vector
std::vector<double> update_graph_weights(const std::vector<double>& drift_probs) {
    std::vector<double> new_weights;
    new_weights.reserve(drift_probs.size());
    
    for (double p : drift_probs) {
        new_weights.push_back(weight_from_probability(p));
    }
    return new_weights;
}

PYBIND11_MODULE(pymatching_binder, m) {
    m.doc() = "C++ accelerator for dynamic graph reweighting in PyMatching";
    
    m.def("update_graph_weights", &update_graph_weights, 
          "Calculates MWPM weights from raw drift probabilities");
}