#include <cassert>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ransac.h"

#include "allan_deviation.h"

#include "imu_data.h"

using namespace allan;

namespace {

constexpr double g = 9.81;
constexpr double pi = 3.14159265358979;

} // namespace

int main(int argc, char *argv[]) {
    assert(argc > 2);
    std::string path = argv[1];
    std::string sensor = argv[2];

    std::vector<imu_reading> imus;
    std::vector<std::pair<imu_reading, double>> imu_stddev;

    std::cout << "\nReading IMU files ...\n";
#ifdef TIMER
    TIMER_BEG(t_loading);
#endif

    ImuCsv imucsv;
    imucsv.load(path);
    imus = std::move(imucsv.items);

#ifdef TIMER
    TIMER_END(t_loading, "loading");
#endif

    std::cout << "\nTotal IMU frames: " << imus.size() << "\n"
              << "\nProcessing IMU readings ...\n"
              << std::scientific << std::setprecision(2);

    /*
     *  start the computation
     */
    double real_tau = (imus.back()[0] - imus.front()[0]) / (imus.size() - 1);

    std::vector<std::pair<size_t, double>> taus;
    for (size_t bucket_size = 10; bucket_size * 2 <= imus.size(); bucket_size = (int)(bucket_size * 1.1)) {
        taus.emplace_back(bucket_size, bucket_size * real_tau);
    }

    for (size_t i = 0; i < taus.size();) {
        std::vector<std::thread> workers;
        size_t threads = std::thread::hardware_concurrency();
        for (size_t j = 0; j < threads && i < taus.size(); ++j, ++i) {
            workers.emplace_back(standard_deviation, imus, taus[i].first, taus[i].second, &imu_stddev);
        }
        for (auto &t : workers) {
            t.join();
        }
    }

    /*
     *  draw plot
     */
    std::vector<std::vector<std::array<double, 3>>> allan_data(imus[0].size() - 1); // size: 3
    for (auto &p : imu_stddev) {
        // p.first[0]: tau
        // p.first[1..3]: stddev
        // p.second: bucket_size
        for (size_t i = 1; i <= allan_data.size(); ++i) {
            allan_data[i - 1].push_back({p.first[0], p.first[i], p.first[i] / sqrt(p.second + 1)});
        }
    }

    white_noise wmodel;
    random_walk rwmodel;
    std::vector<double> crws(allan_data.size());
    std::vector<double> cwns(allan_data.size());

    for (size_t i = 0; i < allan_data.size(); ++i) {
        rwmodel.m_err = wmodel.m_err = allan_data[i][0][2];
        ransac<white_noise>(wmodel, allan_data[i], 0.99);
        ransac<random_walk>(rwmodel, allan_data[i], 0.99);
        cwns[i] = std::pow(10, wmodel.m_noise);
        crws[i] = std::pow(10, rwmodel.m_noise) * std::sqrt(3);
    }

    std::cout << std::setprecision(6)
              << "\nReal frequency: "
              << (imus.size() - 1) / (imus.back()[0] - imus.front()[0])
              << "Hz\nContinuous white noise density:"
              << std::setprecision(9);
    for (auto v : cwns) {
        std::cout << " " << v;
    }
    std::cout << "\nContinuous random walk density:";
    for (auto v : crws) {
        std::cout << " " << v;
    }
    std::cout << "\n";

    gnuplot plot;
    plot.open();

    for (size_t i = 0; i < allan_data.size(); ++i) {
        plot.command("$dim" + std::to_string(i) + " << EOD");
        for (auto p : allan_data[i]) {
            plot.command(std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]));
        }
        plot.command("EOD");
    }

    plot.command("set terminal qt enhanced");
    plot.command("set title '" + sensor + " noise analysis'");
    plot.command("set ylabel 'ADEV'");
    plot.command("set xlabel 'tau'");
    plot.command("set logscale xy");
    plot.command("f(x) = a/sqrt(x)+b*sqrt(x)");

    for (size_t i = 0; i < allan_data.size(); ++i) {
        if (i) {
            plot.command("replot $dim" + std::to_string(i) + " using 1:2:3 with errorbars title 'ADEV'");
        } else {
            plot.command("plot $dim" + std::to_string(i) + " using 1:2:3 with errorbars title 'ADEV'");
        }
        plot.command("replot " + std::to_string(cwns[i]) + " / sqrt(x) title 'dim" + std::to_string(i) + " white noise'");
        plot.command("replot " + std::to_string(crws[i]) + "*sqrt(x) title 'dim" + std::to_string(i) + " random walk'");
    }
    plot.command("pause mouse");
    plot.command("exit");
    plot.close();

    return 0;
}
