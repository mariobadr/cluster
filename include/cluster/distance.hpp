#ifndef CAPPA_CLUSTER_DISTANCE_HPP
#define CAPPA_CLUSTER_DISTANCE_HPP

#include <Eigen/Dense>

namespace cluster {

double euclidean_distance(Eigen::VectorXd const &vector1, Eigen::VectorXd const &vector2);

Eigen::MatrixXd calculate_distance_matrix(Eigen::MatrixXd const &matrix);
}

#endif //CAPPA_CLUSTER_DISTANCE_HPP
