#ifndef CLUPP_PAM_HPP
#define CLUPP_PAM_HPP

#include <Eigen/Dense>

namespace clupp {

// TODO: produce a result as the return type
void partition_around_medoids(int k, Eigen::MatrixXd const &matrix);
}

#endif //CLUPP_PAM_HPP
