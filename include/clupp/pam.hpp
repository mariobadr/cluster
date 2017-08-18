#ifndef CLUPP_PAM_HPP
#define CLUPP_PAM_HPP

#include <Eigen/Dense>

#include <set>
#include <vector>

namespace clupp {

struct pam_result {
  /**
   * The set of row indices that were found to be medoids.
   */
  std::set<int> medoids;

  /**
   * The medoid (cluster) each row was assigned to.
   */
  std::vector<int> classification;
};

pam_result partition_around_medoids(int k, Eigen::MatrixXd const &matrix);
}

#endif //CLUPP_PAM_HPP
