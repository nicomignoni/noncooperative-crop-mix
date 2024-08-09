using Printf, Random, LinearAlgebra, CSV, DataFrames, Tables, Convex, ECOS, PyPlot, LaTeXStrings

na = [CartesianIndex()];

# PyPlots defaults
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["grid.alpha"] = 0.3
rcParams["axes.spines.right"] = false
rcParams["axes.spines.top"] = false
rcParams["legend.frameon"] = false
rcParams["ytick.labelsize"] = 6
rcParams["xtick.labelsize"] = 6
rcParams["font.size"] = 9
rcParams["font.family"] = "sans"
rcParams["font.sans-serif"] = ["Computer Modern Roman"]
rcParams["text.usetex"] = true
rcParams["text.latex.preamble"] = "\\usepackage{amsmath}";
#     raw"\usepackage{amsfonts}", 
#     raw"\usepackage{amssymb}",
# ]

# Read crops data and prices
crop_prices = CSV.read("data/crop-prices.csv", DataFrame);
crop_data   = CSV.read("data/crop-data.csv", DataFrame);
sort!(crop_data, [:Crop, :Technology]);

# Convert Land_Use [tons_per_ha] -> [ha per kg] 
crop_data[!,"Land_Use_ha_per_kg"] = 1 ./ (1e3crop_data[:,"Land_Use_tons_per_ha"]);
select!(crop_data, Not([:Land_Use_tons_per_ha]))

P, T  = length(unique(crop_data.Crop)), length(unique(crop_data.Technology));

# Plot prices
fig, ax = plt.subplots(figsize=(3.5, 1.5))
ax.barh(crop_prices.Crop, crop_prices.Average_Price_per_kg_USD, color="tab:green")
ax.set_xlabel(L"\text{Average price (\$)}", fontsize=8)
ax.grid()

plt.savefig("plot/crop-prices.pdf", bbox_inches="tight")
plt.show()

function heatmap(data, cbar_label, save_path)
    fig, ax = plt.subplots(figsize=(3.5,3))
    hm = ax.imshow(data', cmap="summer")
    
    cbar = ax.figure.colorbar(hm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.99)
    cbar.ax.set_xlabel(cbar_label, fontsize=8)
    
    ax.set_xticks(0:P-1, labels=unique(crop_data.Crop), fontsize=8)
    ax.set_yticks(0:T-1, labels=unique(crop_data.Technology), fontsize=8)
    ax.tick_params(top=true, bottom=false, labeltop=true, labelbottom=false)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
    
    for spine in ["left" "right" "top" "bottom"]
        ax.spines[spine].set_visible(false)
    end
    ax.set_xticks(collect(0:P) .- 0.5, minor=true)
    ax.set_yticks(collect(0:T) .- 0.5, minor=true)
    ax.grid(which="minor", color="white", linewidth=2, alpha=1)
    ax.tick_params(which="minor", bottom=false, left=false)
    
    threshold = (maximum(data) + minimum(data)) / 2
    textcolors=["black", "white"]
    kw = Dict("horizontalalignment" => "center", "verticalalignment" => "center")
    
    for j in 1:P, t in 1:T
        kw["color"] = data[j, t] > threshold ? textcolors[1] : textcolors[2]
        hm.axes.text(j-1, t-1, @sprintf("%.2f", data[j, t]), fontsize=6, kw)
    end
    
    plt.savefig(save_path, bbox_inches="tight")
    plt.show(block=false)
end

crop_data[!,"Total"] = sum(c -> c ./ maximum(c), eachcol(crop_data[:,3:end]))
total_normalized_resource = reshape(crop_data.Total, T, P)'
heatmap(total_normalized_resource, L"$\rho_{ij}$", "plot/crop-data.pdf")

# Custom vectorization
vect(x) = reshape(permutedims(x, [1; collect(ndims(x):-1:2)]), size(x, 1), :);

# Best response
function best_response(g, A, b)
    x = Variable(size(g, 1))
    prob = minimize(g' * x, [A*x <= b])
    solve!(prob, ECOS.Optimizer; silent=true)
    return x.value
end

# Projection operator
function proj(z, A, b)
    x = Variable(size(A, 2))
    prob = minimize(sumsquares(x - z), [A*x <= b])
    solve!(prob, ECOS.Optimizer; silent=true)
    return x.value
end

# Forward-backward-forward
function FBF(x, λ, α)
    N   = size(x, 1)
    x⁺  = stack(i -> proj(x[i,:] - α*(g[i,:] + C[i,:,:]'*λ), A[i,:,:], b[i,:,:]), 1:N, dims=1)
    λ⁺  = max.(0, λ + α*sum(i -> C[i,:,:]*x[i,:], 1:N))
    x⁺⁺ = x⁺ - stack(i -> α*(g[i,:] + C[i,:,:]'*(λ⁺ - λ)), 1:N, dims=1)
    λ⁺⁺ = λ⁺ - α*sum(i -> C[i,:,:]*(x⁺[i,:] - x[i,:]), 1:N)
    return x⁺⁺, λ⁺⁺
end

N = 50 # Num of agents
K = 100 # Num of iterations

Random.seed!(2024)

# Problem data
ℓ = vect(stack(i -> rand(P,T) .+ reshape(crop_data.Land_Use_ha_per_kg, T, P)', 1:N, dims=1)) # land usage
ω = vect(stack(i -> rand(P,T) .+ reshape(crop_data.Water_Use_Liters_per_kg, T, P)', 1:N, dims=1)) # water usage
γ = vect(stack(i -> rand(P,T) .+ reshape(crop_data.CO2_Emissions_kg_per_kg, T, P)', 1:N, dims=1)) # emission
ζ = vect(stack(i -> rand(P,T) .+ reshape(crop_data.Energy_Consumption_MJ_per_kg, T, P)', 1:N, dims=1)) # energy consumption
θ = vect(1e-4rand(N,P) .* crop_prices.Average_Price_per_kg_USD') # unitary price
β = vect(kron(0.8θ, ones(T)'))  # production cost

# Global capacities
ℓₘₐₓ = maximum(ℓ) # max land usage
ωₘₐₓ = 1e-2maximum(ω) # max water usage
γₘₐₓ = maximum(γ) # max emission level

# Local capacities
ζₘₐₓ = 100rand(N) # max energy consumption
βₘₐₓ = 100rand(N) # max budget
θₘᵢₙ = zeros(N)   # minimum return

# Problem parameters
g = β - kron(θ, ones(T)');
A = stack(i -> [-I; β[i,:]'; ζ[i,:]'; -kron(θ[i,:]', ones(T)')], 1:N, dims=1);
C = stack(i -> [ℓ[i,:]'; ω[i,:]'; γ[i,:]'], 1:N, dims=1);
b = stack(i -> [zeros(P*T); βₘₐₓ[i]; ζₘₐₓ[i]; -θₘᵢₙ[i]], 1:N, dims=1);
d = [ℓₘₐₓ; ωₘₐₓ; γₘₐₓ];

# Initial solution
x₀ = stack(i -> best_response(g[i,:], A[i,:,:], b[i,:]), 1:N, dims=1);

# Iterate
α = 1e-5
λ = zeros(K+1, 3)
x = vcat(x₀[na,:,:], zeros(K, N, P*T))
for k = 2:K+1
    x[k,:,:], λ[k,:] = FBF(x[k-1,:,:], λ[k-1,:], α)
end

# Residuals
ρ(k) = sum(i -> C[i,:,:] * x[k,i,:] - d, 1:N);
ρₘₐₓ = stack(k -> maximum(ρ(k)), 1:K);
r    = stack(k -> ρ(k), 1:K);

# Plot multipliers
fig, ax = plt.subplots(figsize=(3.5,2))

ax.plot(
    log.(λ), 
    label=[L"Land usage ($\lambda_1$)", L"Water usage ($\lambda_2$)", L"Emissions ($\lambda_3$)"]
)
ax.set_xlabel(L"Iterations ($\tau$)")
ax.set_ylabel(L"$\log(\lambda)$")
ax.grid()

ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
plt.savefig("plot/lambda.pdf", bbox_inches="tight")
plt.show(block=false)

# Plot residual
fig, ax = plt.subplots(figsize=(3.5,2))

ax.plot(ρₘₐₓ)
ax.set_xlabel(L"Iterations ($\tau$)")
ax.set_ylabel(L"$\max \left\{\sum_{i \in \mathcal{N}} \mathbf{C}_i \mathbf{x}_i - \mathbf{d}\right\}$")
ax.grid()

plt.savefig("plot/residual.pdf", bbox_inches="tight")
plt.show(block=false)

# Save results
avg_x = reshape(sum(i -> x[end,i,:], 1:N), T, P)' ./ N;
CSV.write("result/avg-mix.csv", Tables.table(avg_x), writeheader=false);

heatmap(avg_x, L"$\displaystyle \frac{1}{N} \sum_{i \in \mathcal{N}} x_{ijt}$", "plot/avg-mix.pdf")

