#ifndef CAPPA_CLUSTER_PAM_HPP
#define CAPPA_CLUSTER_PAM_HPP

#include <Eigen/Dense>

#include <map>
#include <set>
#include <vector>

namespace cluster {

/**
 * The clustering result after partitioning around medoids.
 */
struct pam_result {
  /**
   * The objects that were found to be medoids.
   */
  std::set<int> medoids;

  /**
   * The cluster ID each medoid was mapped to.
   */
  std::map<int, int> medoid_to_cluster;

  /**
   * The cluster ID each object was assigned to by the algorithm.
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

#endif //CAPPA_CLUSTER_PAM_HPP
