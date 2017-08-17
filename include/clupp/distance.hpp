#ifndef CLUPP_DISTANCE_HPP
#define CLUPP_DISTANCE_HPP

#include <Eigen/Dense>

namespace clupp {

double euclidean_distance(Eigen::VectorXd const &vector1, Eigen::VectorXd const &vector2);

Eigen::MatrixXd calculate_distance_matrix(Eigen::MatrixXd const &matrix);
}

#endif //CLUPP_DISTANCE_HPP
