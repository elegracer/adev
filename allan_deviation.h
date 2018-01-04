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

typedef std::vector<double> imu_reading;

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

  bool consensus(const point_type &pt) {
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

  bool consensus(const point_type &pt) {
    return std::abs(std::pow(10.0, m_noise + std::log10(pt[0]) * 0.5) - pt[1]) <= m_err;
  }

  static const int n_fit = 1;

  double m_err;
  double m_noise;
};

void standard_deviation(const std::vector<imu_reading>& imus,
                        const double tau,
                        const bool sliding_window,
                        std::vector<std::pair<imu_reading, double>>* imu_stddev) {
#ifdef TIMER
  TIMER_BEG(t_tau);
#endif

  /************************************************************
   * for buckets of period tau compute the mean and stddev
   ************************************************************/

  std::vector<imu_reading> avg_of_buckets;
  imu_reading sum = imus[0];

  if (sliding_window) {
    for (size_t head = 0, tail = 1; tail < imus.size(); ++tail) {
      for (size_t i = 1; i < sum.size(); ++i)
        sum[i] += imus[tail][i];
      double this_dt = std::abs(imus[tail][0] - imus[head][0] - tau);
      double next_dt = tail < imus.size() - 1 ? std::abs(imus[tail + 1][0] - imus[head][0] - tau) : 0;
      if (tail == imus.size()  - 1 || this_dt <= next_dt) {
        auto& back = avg_of_buckets.emplace_back(sum);
        back[0] = imus[tail][0] - imus[head][0]; 
        double count = tail - head + 1;
        for (size_t i = 1; i < back.size(); ++i) {
          back[i] /= count;
          sum[i] -= imus[head][i];
        }
        ++head;
      }
    }
  } else {
    for (size_t i = 1, count = 1; i < imus.size(); ++i) {
      for (size_t j = 1; j < sum.size(); ++j)
        sum[j] += imus[i][j];
      ++count;
      double this_dt = std::abs(imus[i][0] - sum[0] - tau);
      double next_dt = i < imus.size() - 1 ? std::abs(imus[i + 1][0] - sum[0] - tau) : 0;
      if (i == imus.size() - 1 || this_dt <= next_dt) {
        auto& back = avg_of_buckets.emplace_back(sum);
        back[0] = imus[i][0] - sum[0];
        for (size_t j = 1; j < back.size(); ++j)
          back[j] /= count;
        sum[0] = imus[i][0];
        for (size_t j = 1; j < sum.size(); ++j)
          sum[j] = 0;
        count = 0;
      }
    }
  }

  imu_reading mean(imus.front().size(), 0);
  imu_reading stddev(imus.front().size(), 0);

  // compute the mean
  for (size_t i = 0; i < avg_of_buckets.size(); ++i)
    for (size_t j = 0; j < mean.size(); ++j)
      mean[j] += avg_of_buckets[i][j];
  for (size_t i = 0; i < mean.size(); ++i)
    mean[i] /= avg_of_buckets.size();

  // compute the stddev
  stddev[0] = mean[0];
  for (auto& avg_reading : avg_of_buckets)
    for (size_t i = 1; i < stddev.size(); ++i)
      stddev[i] += std::pow(avg_reading[i] - mean[i], 2);
  for (size_t i = 1; i < stddev.size(); ++i)
    stddev[i] = std::sqrt(stddev[i] / (avg_of_buckets.size() - 1.0));

  /************************************************************
   * output & compute Allan Deviation
   ************************************************************/

  std::lock_guard<std::mutex> guard(mutex_for_allan);

  imu_stddev->emplace_back(stddev, avg_of_buckets.size());

#ifdef TIMER
  TIMER_END(t_tau, "[tau=" << tau << "s] size=" << avg_of_buckets.size() << ";");
#endif
  std::cout << "; real_tau=" << mean[0] << "; mean=[ ";
  for (size_t i = 1; i < mean.size(); ++i)
    std::cout << mean[i] << " ";
  std::cout << "]; stddev=[ ";
  for (size_t i = 1; i < stddev.size(); ++i)
    std::cout << stddev[i] << " ";
  std::cout << "]\n" << std::flush;
}

}
