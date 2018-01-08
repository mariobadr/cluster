#include <cluster/pam.hpp>

#include <iostream>

int main(int argc, char **argv)
{
  if(argc != 2) {
    std::cout << "Missing Argument: please enter the number of clusters to create.\n";
    return EXIT_SUCCESS;
  }

  auto const k = static_cast<int>(std::strtol(argv[1], nullptr, 10));

  // build a 1-dimensional matrix of objects
  Eigen::VectorXd data(8);
  data << 942, 2633, 2654, 2137, 373, 434, 1495, 1230;

  // group the data into k clusters
  auto const result = cluster::partition_around_medoids(k, data);

  std::cout << "Medoids:\n";
  for(auto const &medoid : result.medoids) {
    std::cout << medoid << " given ID C" << result.medoid_to_cluster.at(medoid) << "\n";
  }

  std::cout << "\nGrouping:\n";
  for(int i = 0; i < data.rows(); ++i) {
    std::cout << data(i) << " assigned to C" << result.classification[i] << "\n";
  }

  return EXIT_SUCCESS;
}