using PostForecasts
using Statistics
using ProgressBars
using HDF5
import PostForecasts.saveforecasts

# Paths
input_folder = joinpath("data", "processed", "postforecasts", "fARX")
output_folder = joinpath("data", "output", "postforecasts", "fARX")
mkpath(output_folder)

# PostForecasts parameters
calibration_windows = [28, 56, 91, 182] # days
methods = [:zeronormal, :normal, :hs, :qr, :cp, :idr]
start = 20240701
stop = 20241231
quantiles = 99

qf = Dict(m => Vector{QuantForecasts}(undef, 24) for m in methods)

for window in calibration_windows

    println("\n ---- Starting quantile forecasts for calibration window $window ----\n")
    window_folder = joinpath(output_folder, "$(window)D")
    mkpath(window_folder)

    for h in ProgressBar(1:24)
        file = joinpath(input_folder, "h$(h-1).csv")
        pf = loaddlm(file; delim=',', idcol=1, obscol=2, predcol=[3], colnames=true)

        for m in methods
            method_folder = joinpath(window_folder, string(m))
            mkpath(method_folder)

            qf[m][h] = point2quant(pf; method=m, window=window, quantiles=quantiles, start=start, stop=stop)

            filepath = joinpath(method_folder, "h$(h).quantf")
            saveforecasts(qf[m][h], filepath)
        end
    end

    println("Method \t| Avg. CRPS")
    println("-"^25)

    for m in methods
        crps_values = [mean(crps(qf[m][h])) for h in 1:24]
        avg_crps = mean(crps_values)
        println(uppercase(string(m)), "\t| ", round(avg_crps, digits=3))
    end

end