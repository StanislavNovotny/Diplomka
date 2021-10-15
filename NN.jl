include("Funkce.jl")

using Flux
using Plots
using NeuralArithmetic
using LinearAlgebra

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
sqnorm(x) = sum(abs, x)
w_X = 1 ./(1 .+ abs.(X))
w_X_sum = sum(w_X)
loss(x,y,λ) = 1/length(Y).*sum((model(x) - y).^2)  + λ*sum(sqnorm, ps)

max_iter = 6000

#loss(x,y,λ) = sum(w_X.*(model(x) - y).^2) / w_X_sum

η = 0.05
opt = ADAM(η)
LL = zeros(max_iter)

"""
    Model
"""
model = Chain(NaiveNPU(1,1),Dense(1,1,identity))

"""
    Inicializace
"""
Wr = T.([0])[:]
Wi = T.([0])[:]
A = T.([0])
b = T.([0])

λ = 0

init = [(1,1,Wr),(1,2,Wi),(2,1,A), (2,2,b)]
frz = [(1,2)]

ps = params(model)

[init_params!(model, data) for data in init]
[freeze_params!(model, ps, data) for data in frz]

"""
    Iterace
"""

for i=1:max_iter
  LL[i] = loss(X,Y,λ)
  gs = gradient(()->loss(X,Y,λ),ps)
  Flux.Optimise.update!(opt, ps, gs)
end

ps = params(model)
gs = gradient(()->loss(X,Y,λ),ps)

"""
    Vypis parametru
"""
| = Vypis(ps,gs)


"""
    Vykresleni
"""
y = model(X)

scatter(X[:],Y[:],markersize = 1,label="Label")
plot!(X[:],y[:],label="Predikce")

plot(LL, label="Loss function")
plot(log.(LL), label="Log of Loss function")

"""
Vykresleni oblasti k prikladu
"""

x = T.(-0.2:0.1:0.9)
y = T.(-0.2:0.1:0.9)
z = zeros(length(x),length(y))

for i=1:length(x)
    for j=1:length(y)
        tmp = [(1,1,Wr),(1,2,Wi),(2,1,x[i]),(2,2,y[j])]
        [init_params!(model, data) for data in tmp]
        gs = gradient(()->loss(X,Y,λ),ps)
        z[i,j] = norm(gs[ps[1]]) + norm(gs[ps[2]]) + norm(gs[ps[3]]) + norm(gs[ps[4]])
    end
end

contourf(x,y,z,
title = "Velikost norem gradientu",
xlabel = "parametr A",
ylabel = "parametr b",
color = cgrad([:white,:blue]),
contour_labels = true)

