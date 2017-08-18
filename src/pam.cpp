#include "clupp/pam.hpp"

#include "clupp/distance.hpp"

namespace clupp {

pam_result build(int const k, Eigen::MatrixXd const &matrix)
{
  Eigen::MatrixXd const distances = calculate_distance_matrix(matrix);

  // select an initial medoid by finding the observation with the minimum sum of dissimilarities
  int initial_medoid = 0;
  {
    Eigen::VectorXd sum_of_dissimilarities(matrix.rows());

    for(int i = 0; i < distances.rows(); ++i) {
      sum_of_dissimilarities(i) = distances.row(i).sum() + distances.col(i).sum();
    }

    sum_of_dissimilarities.minCoeff(&initial_medoid);
  }

  pam_result initial_clustering;
  initial_clustering.medoids.insert(initial_medoid);
  initial_clustering.classification.resize(static_cast<size_t>(matrix.rows()), initial_medoid);

  // TODO: find additional medoids until k medoids have been found
}

pam_result partition_around_medoids(int k, Eigen::MatrixXd const &matrix)
{
  if(k < 2) {
    throw std::runtime_error("Error: less than two partitions were requested.");
  } else if(matrix.rows() < k) {
    throw std::runtime_error("Error: not enough rows to create k partitions.");
  }

  auto initial_clustering = build(k, matrix);
  // TODO: perform swap phase on initial clustering

  return initial_clustering;
}
}