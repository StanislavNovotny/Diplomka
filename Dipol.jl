include("Funkce.jl")

using MAT
using Flux
using Plots
using NeuralArithmetic
using LinearAlgebra

T = Float32
M = matread("function_data-1D.mat")


"""
    Label
"""
X = Matrix(T.(1.0:220.0)'.-103)/100
Y = Matrix(T.(M["Bx_rez"]))

"""
    Loss function
"""
sqnorm(x) = sum(abs, x)
w_X = 1 ./(1 .+ abs.(X))
w_X_sum = sum(w_X)

max_iter = 7000

λ = 0

loss(x,y,λ) = sum(w_X.*(model(x) - y).^2) / w_X_sum + λ*sum(sqnorm, ps)

η = 0.02
opt = ADAM(η)
LL = zeros(max_iter)

"""
    Model
"""
model = Chain(NaiveNPU(1,2),Dense(2,2,identity),NaiveNPU(2,1))

"""
    Inicializace
"""
Wr1 = T.([1;2])[:]
Wi1 = T.([0;0])[:]
A = T.([16.36 0; 0 1])
b = T.([0,0.15])
Wr2 = T.([1 -2])
Wi2 = T.([0 0])

init = [(1,1,Wr1),(1,2,Wi1),(2,1,A),(2,2,b),(3,1,Wr2),(3,2,Wi2)]
frz = [(1,2),(3,2)]

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
| = Vypis1(ps,gs)


"""
    Vykresleni
"""
y = model(X)

scatter(X[:],Y[:],markersize = 1,label="Label")
plot!(X[:],y[:],label="Predikce")

plot(LL, label="Loss function")
plot(log.(LL), label="Log of Loss function")

