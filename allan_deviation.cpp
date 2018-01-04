#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <experimental/filesystem>

#include "RANSAC.h"

#include "allan_deviation.h"

using namespace std;
using namespace allan;

namespace {

const double g = 9.81;
const double pi = 3.14159265358979;

}

namespace fs = experimental::filesystem;

int main(int argc, char* argv[]) {
  vector<imu_reading> imus;
  vector<pair<imu_reading, double>> imu_stddev;
  string path = argv[1];
  bool sliding_window = false;
  if (argc > 2)
    sliding_window = std::stoi(argv[2]);

  /************************************************************
   * load files
   ************************************************************/

  regex rex(".*gyro_([0-9]+)-([0-9]+)-([0-9]+) ([0-9]+):([0-9]+):([0-9]+)\\.txt");
  smatch match;

  list<pair<string, string>> files;
  for (auto& p: fs::directory_iterator(path)) {
    string filename = p.path().string();
    if (!regex_match(filename, match, rex))
      continue;

    string fid = string(match[1]) + string(match[2]) + string(match[3]) +
                 string(match[4]) + string(match[5]) + string(match[6]);

    auto it = files.begin();
    for (; it != files.end() && it->first < fid; ++it);
    files.insert(it, make_pair(fid, filename));
  }

  cout << "\nReading IMU files ...\n";

#ifdef TIMER
  TIMER_BEG(t_loading);
#endif

  for (auto& file : files) {
    cout << "  [" << file.first << "] \"" << file.second << "\"\n";
    ifstream ifs(file.second);
    string line;
    while (getline(ifs, line)) {
      istringstream iss(line);
      auto& imu = imus.emplace_back();
      size_t i = 0;
      for (double v = 0; iss >> v;) {
        imu.push_back(v);
        ++i;
      }
      if (i < imus.front().size()) {
        imus.pop_back();
        break;
      }
      imu[0] /= 1e9;
    }
    ifs.close();
  }

#ifdef TIMER
  TIMER_END(t_loading, "loading");
#endif

  cout << "\nTotal IMU frames: " << imus.size() << "\n"
       << "\nProcessing IMU readings ...\n"
       << scientific << setprecision(2);

  /************************************************************
   * start the computation
   ************************************************************/

  vector<double> taus;
  for (double tau = 0.1; tau < pow(10, 5); tau *= 1.1) {
    if (sliding_window && imus[imus.size() - 2][0] - imus[0][0] < tau)
      break;
    else if (!sliding_window && imus[imus.size() / 2][0] - imus[0][0] < tau)
      break;
    taus.push_back(tau);
  }

  for (size_t i = 0; i < taus.size();) {
    vector<thread> workers;
    size_t threads = thread::hardware_concurrency();
    for (size_t j = 0; j < threads && i < taus.size(); ++j, ++i)
      workers.emplace_back(standard_deviation, imus, taus[i], sliding_window, &imu_stddev);
    for (auto& t : workers)
      t.join();
  }

  /************************************************************
   * draw plot
   ************************************************************/

  vector<vector<array<double, 3>>> allan_data(imus.front().size() - 1);
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
  plot.command("set title 'Noise Analysis'");
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
