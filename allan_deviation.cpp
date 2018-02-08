#include <cassert>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "RANSAC.h"

#include "allan_deviation.h"

using namespace std;
using namespace allan;

namespace {

constexpr double g = 9.805;
constexpr double pi = 3.14159265358979;

}

int main(int argc, char* argv[]) {
  assert(argc > 2);
  vector<imu_reading> imus;
  vector<pair<imu_reading, double>> imu_stddev;
  string filename = argv[1];
  string sensor = argv[2];
  size_t jump_header = 0;
  if (argc > 3)
    jump_header = stoi(argv[3]);

  /************************************************************
   * load files
   ************************************************************/

  cout << "\nReading IMU files ...\n";

#ifdef TIMER
  TIMER_BEG(t_loading);
#endif

  cout << "\"" << filename << "\"\n";
  ifstream ifs(filename);
  string line;
  while (jump_header) {
    getline(ifs, line);
    --jump_header;
  }
  while (getline(ifs, line)) {
    istringstream iss(line);
    auto& imu = imus.emplace_back();
    size_t i = 0;
    for (double v = 0; iss >> v; imu[i++] = v);
    if (sensor == "acce")
      for (size_t j = 1; j < imu.size(); ++j)
        imu[j] *= g;
    if (i < imus[0].size()) {
      imus.pop_back();
      break;
    }
  }
  ifs.close();

#ifdef TIMER
  TIMER_END(t_loading, "loading");
#endif

  cout << "\nTotal IMU frames: " << imus.size() << "\n"
       << "\nProcessing IMU readings ...\n"
       << scientific << setprecision(2);

  /************************************************************
   * start the computation
   ************************************************************/

  vector<pair<size_t, double>> taus;
  double real_tau = (imus.back()[0] - imus.front()[0]) / (imus.size() - 1);
  for (size_t bucket_size = 10; bucket_size * 2 <= imus.size(); bucket_size = (int)(bucket_size * 1.1))
    taus.emplace_back(bucket_size, bucket_size * real_tau);

  for (size_t i = 0; i < taus.size();) {
    vector<thread> workers;
    size_t threads = thread::hardware_concurrency();
    for (size_t j = 0; j < threads && i < taus.size(); ++j, ++i)
      workers.emplace_back(standard_deviation, imus, taus[i].first, taus[i].second, &imu_stddev);
    for (auto& t : workers)
      t.join();
  }

  /************************************************************
   * draw plot
   ************************************************************/

  vector<vector<array<double, 3>>> allan_data(imus[0].size() - 1);
  for (auto& p : imu_stddev)
    for (size_t i = 1; i <= allan_data.size(); ++i)
      allan_data[i - 1].push_back({{p.first[0], p.first[i], p.first[i] / sqrt(p.second + 1)}});

  white_noise wmodel;
  random_walk rwmodel;
  vector<double> crws(allan_data.size());
  vector<double> cwns(allan_data.size());

  for (size_t i = 0; i < allan_data.size(); ++i) {
    rwmodel.m_err = wmodel.m_err = allan_data[i][0][2];
    ransac<white_noise>(wmodel, allan_data[i], 0.99);
    ransac<random_walk>(rwmodel, allan_data[i], 0.99);
    cwns[i] = pow(10, wmodel.m_noise);
    crws[i] = pow(10, rwmodel.m_noise);
  }

  cout << setprecision(6)
       << "\nReal frequency: "
       << (imus.size() - 1) / (imus.back()[0] - imus.front()[0])
       << "Hz\nContinuous white noise density:"
       << setprecision(9);
  for (auto v : cwns)
    cout << " " << v;
  cout << "\nContinuous random walk density:";
  for (auto v : crws)
    cout << " " << v;
  cout << "\n";

  gnuplot plot;
  plot.open();

  for (size_t i = 0; i < allan_data.size(); ++i) {
    plot.command("$dim" + to_string(i) + " << EOD");
    for (auto p : allan_data[i])
      plot.command(to_string(p[0]) + " " + to_string(p[1]) + " " + to_string(p[2]));
    plot.command("EOD");
  }

  plot.command("set terminal qt enhanced");
  plot.command("set title '" + sensor + " noise analysis'");
  plot.command("set ylabel 'ADEV'");
  plot.command("set xlabel 'tau'");
  plot.command("set logscale xy");
  plot.command("f(x) = a/sqrt(x)+b*sqrt(x)");

  for (size_t i = 0; i < allan_data.size(); ++i) {
    if (i)
      plot.command("replot $dim" + to_string(i) + " using 1:2:3 with errorbars title 'ADEV'");
    else
      plot.command("plot $dim" + to_string(i) + " using 1:2:3 with errorbars title 'ADEV'");
    plot.command("replot " + to_string(cwns[i]) + " / sqrt(x) title 'dim" + to_string(i) + " white noise'");
    plot.command("replot " + to_string(crws[i]) + "*sqrt(x) title 'dim" + to_string(i) + " random walk'");
  }
  plot.command("pause mouse");
  plot.command("exit");
  plot.close();

  return 0;
}
