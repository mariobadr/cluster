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
  std::set<int> medoids;

  std::vector<int> classification;
};

pam_result partition_around_medoids(int k, Eigen::MatrixXd const &matrix);
}

#endif //CLUPP_PAM_HPP
