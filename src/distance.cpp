#include "clupp/distance.hpp"

namespace clupp {

double euclidean_distance(Eigen::VectorXd const &vector1, Eigen::VectorXd const &vector2)
{
  // the 2D norm is the euclidean distance
  // see: https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
  return (vector1 - vector2).lpNorm<2>();
}

Eigen::MatrixXd calculate_distance_matrix(Eigen::MatrixXd const &matrix)
{
  // TODO: use a sparse matrix here instead
  Eigen::MatrixXd distance_matrix = Eigen::MatrixXd::Constant(matrix.rows(), matrix.rows(), 0.0);

  for(int i = 0; i < matrix.rows(); ++i) {
    for(int j = i + 1; j < matrix.rows(); ++j) {
      distance_matrix(i, j) = euclidean_distance(matrix.row(i), matrix.row(j));
    }
  }

  return distance_matrix;
}
}