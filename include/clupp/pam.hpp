#ifndef CLUPP_PAM_HPP
#define CLUPP_PAM_HPP

#include <Eigen/Dense>

#include <set>
#include <vector>

namespace clupp {

/**
 * The clustering result after partitioning around medoids.
 */
class pam_result {
public:
  /**
   * Constructor.
   *
   * @param number_of_objects The number of observations (i.e., rows) in the initial data.
   * @param initial_medoid The first medoid all objects are assigned to.
   */
  explicit pam_result(int number_of_objects, int initial_medoid);

  /**
   * Add a new medoid to the result.
   *
   * @param medoid The object that is the new medoid.
   */
  void add_medoid(int medoid);

  /**
   * Assign an object to a medoid.
   *
   * @param object The object to assign.
   * @param medoid The object's new medoid.
   */
  void assign_medoid(int object, int medoid);

  /**
   * @return the set of objects that were not medoids.
   */
  std::set<int> nonselected_objects() const
  {
    return m_nonselected;
  }

  /**
   * Find the medoid assigned to an object.
   *
   * @param object The object to query.
   *
   * @return The medoid the object is assigned to.
   */
  int medoid(int object) const
  {
    return m_classification.at(static_cast<size_t>(object));
  }

private:
  std::set<int> m_medoids;

  std::set<int> m_nonselected;

  std::vector<int> m_classification;
};

pam_result partition_around_medoids(int k, Eigen::MatrixXd const &matrix);
}

#endif //CLUPP_PAM_HPP
