#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <array>

using imu_data = std::array<double, 4>;  // t, x, y, z

struct ImuCsv {
    ImuCsv() {}
    ImuCsv(const std::string type) :
        type(type) {}

    std::vector<imu_data> items;
    std::string type;

    void append(double t, double x, double y, double z)
    {
        items.emplace_back(imu_data {t, x, y, z});
    }

    imu_data mean_value()
    {
        imu_data mean = {0.0, 0.0, 0.0, 0.0};
        for (auto &item : items) {
            for (size_t i = 1; i < item.size(); ++i) {
                mean[i] += item[i];
            }
        }

        mean[0] = (items.size() - 1) / (items.back()[0] - items.front()[0]);
        for (size_t i = 1; i < items.size(); ++i) {
            mean[i] /= items.size();
        }

        return mean;
    }

    void load(const std::string &filename)
    {
        items.clear();
        if (FILE *csv = fopen(filename.c_str(), "r")) {
            char header_line[2048];
            fscanf(csv, "%2047[^\n]", header_line);
            imu_data item;
            while (!feof(csv) && fscanf(csv, "%lf,%lf,%lf,%lf\n", &item[0], &item[1], &item[2], &item[3]) == 4) {
                items.emplace_back(std::move(item));
            }
            fclose(csv);
        }
    }

    void save(const std::string &filename) const
    {
        if (FILE *csv = fopen(filename.c_str(), "w")) {
            if (type == "acc") {
                fputs("#acc: t[s:double],x[m/s^2:double],y[m/s^2:double],z[m/s^2:double]\n", csv);
            } else if (type == "gyr") {
                fputs("#gyr: t[s:double],x[rad/s:double],y[rad/s:double],z[rad/s:double]\n", csv);
            }
            for (auto &item : items) {
                fprintf(csv, "%.14e,%.9e,%.9e,%.9e\n", item[0], item[1], item[2], item[3]);
            }
            fclose(csv);
        }
    }

    void save_column(const std::string &filename, size_t col) const
    {
        if (FILE *csv = fopen(filename.c_str(), "w")) {
            for (auto &item : items) {
                fprintf(csv, "%.9e\n", item[col]);
            }
            fclose(csv);
        }
    }
};