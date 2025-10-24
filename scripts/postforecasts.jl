using PostForecasts
using Statistics

methods = [:zeronormal, :normal, :hs, :qr, :cp, :idr]
qf = Dict(m => Vector{QuantForecasts}(undef, 24) for m in methods)

for h in 1:24
    file = joinpath("data", "processed", "postforecasts", "lear", "h$(h-1).csv")
    pf = loaddlm(file; delim=',', idcol=1, obscol=2, predcol=[3], colnames=true)

    for m in methods
        qf[m][h] = point2quant(pf; method=m, window=182, quantiles=99, start=20240701, stop=20241231)
    end
end

println("Method \t| Avg. CRPS")
println("-"^25)

for m in methods
    crps_values = [mean(crps(qf[m][h])) for h in 1:24]
    avg_crps = mean(crps_values)
    println(uppercase(string(m)), "\t| ", round(avg_crps, digits=3))
end