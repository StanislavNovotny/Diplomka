include("Funkce.jl")

using Flux
using Plots
using NeuralArithmetic
using LinearAlgebra
using JLD

T = Float32

"""
    Label
"""
p(x) = 2*x^2

X = Matrix(T.(-1.0:0.001:1.0)')
Y = p.(X)

"""
    Loss function
"""
λ = 0
sqnorm(x) = sum(abs, x)
w_X = 1 ./(1 .+ abs.(X))
w_X_sum = sum(w_X)
loss(x,y,λ) = 1/length(Y).*sum((model(x) - y).^2)  + λ*sum(sqnorm, ps)

η = 0.05
opt = ADAM(η)

"""
    Model
"""
model = Chain(NaiveNPU(1,1),Dense(1,1,identity))

ps = params(model)

"""
Vykresleni konvergence
"""
r = T.(-1.0:0.1:3.0)
a = T.(-1.0:0.1:3.0)
z = zeros(length(r),length(a))

for i=1:length(r)
    for j=1:length(a)
        tmp = [(1,1,r[i]),(1,2,T.([0])[:]),(2,1,a[j]),(2,2,T.([0]))]
        frz = [(1,2)]
        ps = params(model)
        [init_params!(model, data) for data in tmp]
        [freeze_params!(model, ps, data) for data in frz]
        for k=1:5200
            gs = gradient(()->loss(X,Y,λ),ps)
            Flux.Optimise.update!(opt, ps, gs)
        end
        ps = params(model)
        gs = gradient(()->loss(X,Y,λ),ps)
        if (norm(gs[ps[1]]) + norm(gs[ps[3]]) + norm(gs[ps[4]])) >= 0.025
            z[i,j] = 0
        elseif abs(ps[1][1] - 2) < 0.025 && abs(ps[3][1] - 2) < 0.025 && abs(ps[4][1]) < 0.025
            z[i,j] = -1
        elseif abs(ps[1][1]) < 0.025 && abs(ps[3][1]) < 0.025 && abs(ps[4][1] - 0.66) < 0.025
            z[i,j] = 1
        else z[i,j] = NaN
        end
    end
end

Vysledek1 = z
save("b0.jld", "b0", Vysledek1)

q = zeros(length(r),length(a))

for i=1:length(r)
    for j=1:length(a)
        tmp = [(1,1,r[i]),(1,2,T.([0])[:]),(2,1,a[j]),(2,2,T.([sum(Y)/length(Y)]))]
        frz = [(1,2)]
        ps = params(model)
        [init_params!(model, data) for data in tmp]
        [freeze_params!(model, ps, data) for data in frz]
        for k=1:5200
            gs = gradient(()->loss(X,Y,λ),ps)
            Flux.Optimise.update!(opt, ps, gs)
        end
        ps = params(model)
        gs = gradient(()->loss(X,Y,λ),ps)
        if (norm(gs[ps[1]]) + norm(gs[ps[3]]) + norm(gs[ps[4]])) >= 0.025
            z[i,j] = 0
        elseif abs(ps[1][1] - 2) < 0.025 && abs(ps[3][1] - 2) < 0.025 && abs(ps[4][1]) < 0.025
            z[i,j] = -1
        elseif abs(ps[1][1]) < 0.025 && abs(ps[3][1]) < 0.025 && abs(ps[4][1] - 0.66) < 0.025
            z[i,j] = 1
        else z[i,j] = NaN
        end
    end
end

Vysledek2 = q
save("b666.jld", "b666", Vysledek2)


fig1 = heatmap(r,a, q,
    c=cgrad([:green, :red,]),
    xlabel="Wr", ylabel="A",
    title="Konvergence pro b = 2/3",
    xlims = (-1.1,3.1),
    ylims = (-1.1,3.1),
    minorgrid = true,
    draw_arrow = true,
    colorbar=false,
    minorticks = 10
)


savefig(fig1,"b23conv")

fig2 = heatmap(r,a, z,
c=cgrad([:green, :red,]),
xlabel="Wr", ylabel="A",
title="Konvergence pro b = 0",
xlims = (-1.1,3.1),
ylims = (-1.1,3.1),
minorgrid = true,
draw_arrow = true,
colorbar=false,
minorticks = 10
)

savefig(fig2,"b0conv")
