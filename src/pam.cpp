#include "clupp/pam.hpp"

#include "clupp/distance.hpp"

#include <iostream>

namespace clupp {

/**
 * Data used during the PAM algorithm.
 */
struct pam_data {
  std::set<int> medoids;
  std::set<int> nonselected;
  std::vector<int> classification;
  std::vector<int> second_closest_medoid;
  double total_dissimilarity;

  pam_data(int number_of_objects, int initial_medoid)
      : classification(number_of_objects, initial_medoid)
      , second_closest_medoid(number_of_objects, -1)
      , total_dissimilarity(0.0)
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

  void swap_medoid(int old_medoid, int new_medoid)
  {
    medoids.erase(old_medoid);
    nonselected.insert(old_medoid);

    add_medoid(new_medoid);

    for(auto &medoid : classification) {
      if(medoid == old_medoid) {
        medoid = new_medoid;
      }
    }

    for(auto &medoid : second_closest_medoid) {
      if(medoid == old_medoid) {
        medoid = new_medoid;
      }
    }
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
 * @param clustering The current clustering state.
 *
 * @return The index of the object that was found to be the medoid.
 */
int find_next_medoid(Eigen::MatrixXd const &distances, pam_data const &clustering)
{
  double maximum_gain = std::numeric_limits<double>::lowest();
  int next_medoid = 0;

  // consider an object i which has not been selected yet
  for(auto const i : clustering.nonselected) {
    auto nonselected = clustering.nonselected;
    nonselected.erase(i);

    // track the potential gain of selecting i as a new medoid
    double gain = 0.0;

    // consider another nonselected object j
    for(auto const j : nonselected) {
      // calculate the dissimilarity between j and its currently assigned cluster
      double const D_j = distances(j, clustering.classification[j]);
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
 * @param clustering The clustering state to modify.
 */
void reclassify_objects(Eigen::MatrixXd const &distances, pam_data *clustering)
{
  // reset the total dissimilarity
  clustering->total_dissimilarity = 0.0;

  for(int object = 0; object < distances.rows(); ++object) {
    double closest_distance = std::numeric_limits<double>::max();
    double second_closest_distance = std::numeric_limits<double>::max();

    int closest_medoid = -1;
    int second_closest_medoid = -1;

    for(auto const medoid : clustering->medoids) {
      double const distance = distances(object, medoid);

      if(distance < closest_distance || clustering->classification[medoid] == object) {
        second_closest_distance = closest_distance;
        second_closest_medoid = closest_medoid;

        closest_distance = distance;
        closest_medoid = medoid;
      } else if(distance < second_closest_distance) {
        second_closest_distance = distance;
        second_closest_medoid = medoid;
      }
    }

    clustering->classification[object] = closest_medoid;
    clustering->second_closest_medoid[object] = second_closest_medoid;
    clustering->total_dissimilarity += closest_distance;
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
    initial_clustering.add_medoid(find_next_medoid(distances, initial_clustering));
    reclassify_objects(distances, &initial_clustering);
  }

  return initial_clustering;
}

/**
 * Calculates the effect a swap between i and h will have on the value of the clustering.
 *
 * @param distances The distance matrix.
 * @param i A currently selected medoid.
 * @param h An object that has not been selected as a medoid.
 * @param clustering The current clustering state.
 *
 * @return The total contribution of the swap. A negative value means the swap improves the clustering.
 */
double calculate_swap_cost(Eigen::MatrixXd const &distances,
    int const i,
    int const h,
    pam_data const &clustering)
{
  auto nonselected = clustering.nonselected;
  nonselected.erase(h);

  double total_contribution = 0.0;
  for(auto const j : nonselected) {
    auto const D_j = distances(j, clustering.classification[j]);
    auto const d_j_i = distances(j, i);
    auto const d_j_h = distances(j, h);

    double contribution = 0.0;
    if(D_j >= d_j_i) {
      // j is not further from i than its current (and therefore any other) medoid

      // calculate distance to second closest medoid
      auto const E_j = distances(j, clustering.second_closest_medoid[j]);

      if(d_j_h < E_j) {
        // j is closer to h than the second closest medoid
        // if j is closer to i than h, contribution is positive (swap is not favourable)
        contribution = d_j_h - d_j_i;
      } else {
        // j is at least as distant to h than the second closest medoid
        // contribution is always positive because h is futher away
        contribution = E_j - D_j;
      }
    } else if(D_j < d_j_i && D_j > d_j_h) {
      // j is more distance from i but closer to h
      contribution = d_j_h - D_j;
    }

    // add up all the contributions for the total result of a swap
    total_contribution += contribution;
  }

  return total_contribution;
}

/**
 * Attempt to improve the set of medoids by considering all pairs of objects where a medoid i has been selected but an
 * object h has not, and testing if a swap is beneficial.
 *
 * @param distances The distance matrix.
 * @param clustering The clustering state to improve.
 */
void refine(Eigen::MatrixXd const &distances, pam_data *clustering)
{
  bool perform_swaps = true;

  while(perform_swaps) {
    double minimum_contribution = std::numeric_limits<double>::max();
    int old_medoid = -1;
    int new_medoid = -1;

    for(auto const i : clustering->medoids) {
      for(auto const h : clustering->nonselected) {
        auto const contribution = calculate_swap_cost(distances, i, h, *clustering);

        // minimize the total result of a swap (i.e., most negative contribution)
        if(contribution < minimum_contribution) {
          minimum_contribution = contribution;
          old_medoid = i;
          new_medoid = h;
        }
      }
    }

    if(minimum_contribution < 0 && old_medoid >= 0 && new_medoid >= 0) {
      // if the minimum contribution was negative, perform the swap and iterate again
      clustering->swap_medoid(old_medoid, new_medoid);
      reclassify_objects(distances, clustering);
    } else {
      // a positive minimum contribution means that no swaps were favourable
      perform_swaps = false;
    }
  }
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

  // build an initial clustering based on the minimum dissimilarity between objects
  auto initial_clustering = build(k, distances);

  // refine the initial clustering by swapping medoids and optimizing the objective function
  refine(distances, &initial_clustering);

  // copy the intermediate data into the final result
  pam_result final_clustering;
  final_clustering.medoids = initial_clustering.medoids;

  int cluster_id = 0;
  for(auto const &medoid : final_clustering.medoids) {
    final_clustering.medoid_to_cluster[medoid] = cluster_id;
    ++cluster_id;
  }

  for(auto const &object : initial_clustering.classification) {
    final_clustering.classification.push_back(final_clustering.medoid_to_cluster[object]);
  }

  return final_clustering;
}
}