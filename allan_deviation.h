#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <vector>

#include "gnuplot.h"

#ifndef TIMER
#define TIMER
#include <chrono>
#include <iostream>
#define TIMER_BEG(timer) \
  auto timer = std::chrono::steady_clock::now();
#define TIMER_END(timer, msg) \
  std::cout << "[T] \"" << msg << "\": " \
            << std::chrono::duration<double>( \
                 std::chrono::steady_clock::now() - timer \
               ).count() \
            << "s" << std::flush
#endif

namespace allan {

typedef std::array<double, 4> imu_reading;

std::mutex mutex_for_allan;

class white_noise {
 public:
  typedef std::array<double, 3> point_type;

  bool fit(const std::vector<point_type>& pts,
           const std::vector<unsigned char>& inlier_mask) {
    double new_noise = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        double b = std::log10(pts[i][1]) + std::log10(pts[i][0]) * 0.5;
        new_noise += b;
        ++count;
      }
    }
    m_noise = new_noise / count;
    return true;
  }

  bool consensus(const point_type& pt) {
    return std::abs(std::pow(10.0, m_noise - std::log10(pt[0]) * 0.5) - pt[1]) <= m_err;
  }

  static const int n_fit = 1;

  double m_err;
  double m_noise;
};

class random_walk {
 public:
  typedef std::array<double, 3> point_type;

  bool fit(const std::vector<point_type>& pts,
           const std::vector<unsigned char>& inlier_mask) {
    double new_noise = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        double b = std::log10(pts[i][1]) - std::log10(pts[i][0]) * 0.5;
        new_noise += b;
        ++count;
      }
    }
    m_noise = new_noise / count;
    return true;
  }

  bool consensus(const point_type& pt) {
    return std::abs(std::pow(10.0, m_noise + std::log10(pt[0]) * 0.5) - pt[1]) <= m_err;
  }

  static const int n_fit = 1;

  double m_err;
  double m_noise;
};

void standard_deviation(const std::vector<imu_reading>& imus,
                        const size_t bucket_size,
                        const double tau,
                        std::vector<std::pair<imu_reading, double>>* imu_stddev) {
#ifdef TIMER
  TIMER_BEG(t_tau);
#endif

  /************************************************************
   * for buckets of `bucket_size` compute the mean and stddev
   ************************************************************/

  std::vector<imu_reading> avg_of_buckets;

  // compute the stddev
  imu_reading stddev {{0, 0, 0, 0}};

  for (size_t bucket_id = 0; bucket_id * bucket_size + bucket_size <= imus.size(); ++bucket_id) {
    auto& mean = avg_of_buckets.emplace_back();
    mean[0] = imus[bucket_id * bucket_size + 1][0] - imus[bucket_id * bucket_size][0];
    mean[0] *= bucket_size;
    for (size_t i = 0; i < bucket_size; ++i) {
      const auto& imu = imus[bucket_id * bucket_size + i];
      for (size_t j = 1; j < mean.size(); ++j)
        mean[j] += imu[j];
    }
    for (size_t j = 1; j < mean.size(); ++j)
      mean[j] /= bucket_size;
  }

  stddev[0] = tau;
  for (size_t i = 1; i < avg_of_buckets.size(); ++i) 
    for (size_t j = 1; j < stddev.size(); ++j)
      stddev[j] += std::pow(avg_of_buckets[i][j] - avg_of_buckets[i - 1][j], 2);
  for (size_t i = 1; i < stddev.size(); ++i) {
    stddev[i] /= (avg_of_buckets.size() - 1) * 2;
    stddev[i] = std::sqrt(stddev[i]);
  }

  /************************************************************
   * output & compute Allan Deviation
   ************************************************************/

  std::lock_guard<std::mutex> guard(mutex_for_allan);

  imu_stddev->emplace_back(stddev, avg_of_buckets.size());

#ifdef TIMER
  TIMER_END(t_tau, "[bucket=" << bucket_size << "] size=" << avg_of_buckets.size() << ";");
#endif
  std::cout << "; real_tau=" << stddev[0] << "s; stddev=[ ";
  for (size_t i = 1; i < stddev.size(); ++i)
    std::cout << stddev[i] << " ";
  std::cout << "]\n" << std::flush;
}

}
