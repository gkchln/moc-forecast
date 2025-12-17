using ArgParse
using PostForecasts
using Statistics
using ProgressBars
using HDF5
import PostForecasts.saveforecasts

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input_folder"
            help = "Input folder path"
            arg_type = String
            required = true

        "--output_folder"
            help = "Output folder path"
            arg_type = String
            required = true

        "--calibration_windows"
            help = "Comma-separated list of calibration windows (days)"
            arg_type = String
            required = true

        "--methods"
            help = "Comma-separated list of methods"
            arg_type = String
            required = true

        "--start"
            help = "Start date (YYYYMMDD)"
            arg_type = Int
            required = true

        "--stop"
            help = "Stop date (YYYYMMDD)"
            arg_type = Int
            required = true

        "--quantiles"
            help = "Number of quantiles"
            arg_type = Int
            default = 99
    end

    return parse_args(s)
end

args = parse_commandline()

# Convert arguments
input_folder = args["input_folder"]
output_folder = args["output_folder"]

calibration_windows = parse.(Int, split(args["calibration_windows"], ","))
methods = Symbol.(split(args["methods"], ","))

start = args["start"]
stop = args["stop"]
quantiles = args["quantiles"]

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