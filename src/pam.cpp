#include "clupp/pam.hpp"

#include "clupp/distance.hpp"

namespace clupp {

void build(int const k, Eigen::MatrixXd const &matrix)
{
  Eigen::MatrixXd const distance_matrix = calculate_distance_matrix(matrix);

  // select an initial medoid by finding the observation with the minimum sum of dissimilarities
  int initial_medoid = 0;
  {
    Eigen::VectorXd sum_of_dissimilarities(matrix.rows());

    for (int i = 0; i < distance_matrix.rows(); ++i) {
      sum_of_dissimilarities(i) = distance_matrix.row(i).sum() + distance_matrix.col(i).sum();
    }

    sum_of_dissimilarities.minCoeff(&initial_medoid);
  }

  // TODO: classify all operations as being part of the initial medoid

  // TODO: find additional medoids until k medoids have been found
}

void partition_around_medoids(int const k, Eigen::MatrixXd const &matrix)
{
  if(k < 2) {
    throw std::runtime_error("Error: less than two partitions were requested.");
  }
  // TODO: check we have enough observations in matrix for k clusters

  build(k, matrix);
  // TODO: perform swap phase
}
}