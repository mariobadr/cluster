#ifndef CLUPP_PAM_HPP
#define CLUPP_PAM_HPP

#include <Eigen/Dense>

#include <set>
#include <vector>

namespace clupp {

/**
 * The clustering result after partitioning around medoids.
 */
struct pam_result {
  /**
   * The objects that were found to be medoids.
   */
  std::set<int> medoids;

  /**
   * The medoid each object was assigned to by the clustering algorithm.
   */
  std::vector<int> classification;
};

/**
 * Minimize the sum of dissimilarities to a set of k medoids.
 *
 * @param k The number of clusters.
 * @param matrix The objects observed.
 *
 * @return The clustering found.
 */
pam_result partition_around_medoids(int k, Eigen::MatrixXd const &matrix);
}

#endif //CLUPP_PAM_HPP
