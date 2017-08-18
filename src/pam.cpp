#include "clupp/pam.hpp"

#include "clupp/distance.hpp"

namespace clupp {

/**
 * Data used during the PAM algorithm.
 */
struct pam_data {
  std::set<int> medoids;
  std::set<int> nonselected;
  std::vector<int> classification;
  std::vector<int> second_nearest_medoid;

  pam_data(int number_of_objects, int initial_medoid)
      : classification(number_of_objects, initial_medoid)
      , second_nearest_medoid(number_of_objects, initial_medoid)
  {
    for(int i = 0; i < number_of_objects; ++i) {
      nonselected.insert(nonselected.end(), i);
    }

    medoids.insert(initial_medoid);
    nonselected.erase(initial_medoid);
  }

  void assign_medoid(int object, int medoid)
  {
    classification[object] = medoid;
  }

  void add_medoid(int medoid)
  {
    medoids.insert(medoid);
    nonselected.erase(medoid);

    assign_medoid(medoid, medoid);
  }
};

/**
 * The initial medoid is the object with the minimum sum of dissimilarities to all other objects.
 *
 * @param distances The distance matrix.
 *
 * @return The index of the object that was found to be the medoid.
 */
int find_initial_medoid(Eigen::MatrixXd const &distances)
{
  Eigen::VectorXd sum_of_dissimilarities(distances.rows());

  for(int i = 0; i < distances.rows(); ++i) {
    sum_of_dissimilarities(i) = distances.row(i).sum();
  }

  int initial_medoid;
  sum_of_dissimilarities.minCoeff(&initial_medoid);

  return initial_medoid;
}

/**
 * The next medoid is the the a nonselected object that decreases the objective function the most.
 *
 * @param distances The distance matrix.
 * @param current_clustering The current clustering state.
 *
 * @return The index of the object that was found to be the medoid.
 */
int find_next_medoid(Eigen::MatrixXd const &distances, pam_data const &current_clustering)
{
  double maximum_gain = std::numeric_limits<double>::min();
  int next_medoid = 0;

  // consider an object i which has not been selected yet
  for(auto const i : current_clustering.nonselected) {
    auto nonselected = current_clustering.nonselected;
    nonselected.erase(i);

    // track the potential gain of selecting i as a new medoid
    double gain = 0.0;

    // consider another nonselected object j
    for(auto const j : nonselected) {
      // calculate the dissimilarity between j and its currently assigned cluster
      double const D_j = distances(j, current_clustering.classification[j]);
      // calculate the dissimilarity between j and i
      double const d_j_i = distances(j, i);

      // if the difference of these dissimliarities is positive, it contributes to the selection of i
      gain += std::max(D_j - d_j_i, 0.0);
    }

    // choose the nonselected object that maximizes the gain
    if(gain > maximum_gain) {
      maximum_gain = gain;
      next_medoid = i;
    }
  }

  return next_medoid;
}

/**
 * Reassign objects in the current clustering for the new medoid.
 *
 * @param distances The distance matrix.
 * @param new_medoid The new medoid.
 * @param current_clustering The clustering state to modify.
 */
void reclassify_objects(Eigen::MatrixXd const &distances,
    int const new_medoid,
    pam_data *current_clustering)
{
  current_clustering->add_medoid(new_medoid);

  for(auto const object : current_clustering->nonselected) {
    auto const current_medoid = current_clustering->classification[object];
    auto const second_closest_medoid = current_clustering->second_nearest_medoid[object];

    auto const current_distance = distances(object, current_medoid);
    auto const second_distance = distances(object, second_closest_medoid);
    auto const potential_distance = distances(object, new_medoid);

    if(potential_distance < current_distance) {
      current_clustering->second_nearest_medoid[object] = current_medoid;
      current_clustering->assign_medoid(object, new_medoid);
    } else if(potential_distance < second_distance) {
      current_clustering->second_nearest_medoid[object] = new_medoid;
    }
  }
}

/**
 * The first phase of pam produces an initial clustering for k objects.
 *
 * @param k The number of initial clusters to find.
 * @param matrix The observations.
 *
 * @return An initial clustering of observations to k objects.
 */
pam_data build(int const k, Eigen::MatrixXd const &distances)
{
  // select an initial medoid by finding the observation with the minimum sum of dissimilarities
  int const initial_medoid = find_initial_medoid(distances);

  // create the initial clustering based on the initial medoid
  pam_data initial_clustering(static_cast<int>(distances.rows()), initial_medoid);

  // refine the initial clustering with an additional k - 1 medoids
  for(int i = 0; i < k - 1; ++i) {
    int next_medoid = find_next_medoid(distances, initial_clustering);
    reclassify_objects(distances, next_medoid, &initial_clustering);
  }

  return initial_clustering;
}

void refine(Eigen::MatrixXd const &distances, pam_data *initial_clustering)
{
}

pam_result partition_around_medoids(int k, Eigen::MatrixXd const &matrix)
{
  if(k < 2) {
    throw std::runtime_error("Error: less than two partitions were requested.");
  } else if(matrix.rows() < k) {
    throw std::runtime_error("Error: not enough rows to create k partitions.");
  }

  // calculate the distances between observations
  Eigen::MatrixXd const distances = calculate_distance_matrix(matrix);

  auto initial_clustering = build(k, distances);
  refine(distances, &initial_clustering);

  pam_result final_clustering{};
  final_clustering.medoids = initial_clustering.medoids;
  final_clustering.classification = initial_clustering.classification;

  return final_clustering;
}
}