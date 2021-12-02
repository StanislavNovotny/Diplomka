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
        for k=1:4350
            gs = gradient(()->loss(X,Y,λ),ps)
            Flux.Optimise.update!(opt, ps, gs)
        end
        ps = params(model)
        gs = gradient(()->loss(X,Y,λ),ps)
        z[i,j] = evaluate_solution(ps, gs; ϵ=0.025)
    end
end

save("b0final.jld", "b0final", z)

q = zeros(length(r),length(a))

for i=1:length(r)
    for j=1:length(a)
        tmp = [(1,1,r[i]),(1,2,T.([0])[:]),(2,1,a[j]),(2,2,T.([sum(Y)/length(Y)]))]
        frz = [(1,2)]
        ps = params(model)
        [init_params!(model, data) for data in tmp]
        [freeze_params!(model, ps, data) for data in frz]
        for k=1:4350
            gs = gradient(()->loss(X,Y,λ),ps)
            Flux.Optimise.update!(opt, ps, gs)
        end
        ps = params(model)
        gs = gradient(()->loss(X,Y,λ),ps)
        q[i,j] = evaluate_solution(ps, gs; ϵ=0.025)
    end
end


save("b666final.jld", "b666final", q)


fig1 = heatmap(r,a, q,
    c=cgrad([:yellow,:green, :red, :white]),
    xlabel="W^r", ylabel="A",
    title="Konvergence pro b = 2/3",
    xlims = (-1.1,3.1),
    ylims = (-1.1,3.1),
    minorgrid = true,
    draw_arrow = true,
    colorbar=false,
    minorticks = 10
)


savefig(fig1,"FFinalb23conv")


z = load("C:/Users/stany/b0final.jld", "b0final")
fig2 = heatmap(r,a, z,
c=cgrad([:yellow,:green, :red, :white]),
xlabel="W^r", ylabel="A",
title="Konvergence pro b = 0",
xlims = (-1.1,3.1),
ylims = (-1.1,3.1),
minorgrid = true,
draw_arrow = true,
colorbar=false,
minorticks = 10
)

savefig(fig2,"FFFinalb0conv")
