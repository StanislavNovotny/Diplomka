include("Funkce.jl")

using MAT
using Flux
using Plots
using NeuralArithmetic
using LinearAlgebra
using Statistics

#data
data2d = matread("function_data-2D.mat")
zdata = data2d["Bx"]/abs(maximum(data2d["Bx"]))

plot(collect(1.0:220.0),data2d["Bx"][159,:])

T = Float32
X = Matrix(T.((1.0:220.0)'.-103.5)/100)
XX = T.((ones(length(X),1))*X)
YY = T.(X'*(ones(length(X),1)'))

plot(XX,YY,zdata,st=:surface, color= :blues,xlabel="osa x", ylabel=:"osa y", zlabel="Bx")

sqnorm(x) = sum(abs, x)
w_X = 1 ./(1 .+ abs.(X))
w_X_sum = sum(w_X)

max_iter = 11000

λ = 0

#model
model = Chain(NaiveNPU(1,5),Dense(5,440,identity), NaiveNPU(440,220))

loss(x,y,λ) = sum(w_X.*(model(x) - y).^2) / w_X_sum + λ*sum(sqnorm, ps)

η = 0.004
opt = ADAM(η)
LL = zeros(max_iter)

#init
Wr1 = T.([1;1;2;3;4])
Wi1 = T.(zeros(5,1))

Wr2 = 
Wi2 = T.(zeros(220,440))

init = [(1,1,Wr1),(1,2,Wi1),(3,2,Wi2)]
frz = [(1,2),(3,2)]

ps = Flux.params(model)

[init_params!(model, data) for data in init]
[freeze_params!(model, ps, data) for data in frz]

#Iterace

ps = Flux.params(model)

@time begin
    for i=1:max_iter
        LL[i] = loss(X,zdata,λ)
        gs = gradient(()->loss(X,zdata,λ),ps)
        Flux.Optimise.update!(opt, ps, gs)
        println(LL[i])
        if isnan(LL[i])
            break
        end
        if LL[i] < 1000
            break
        end
    end
end

ps = Flux.params(model)
gs = gradient(()->loss(X,zdata,λ),ps)

| = Vypis1(ps,gs)

rmsdModel = sqrt(mean(abs2,model(X)-zdata))

pyplot()

plot(XX,YY,model(X),st=:surface, color= :blues,xlabel="osa x", ylabel=:"osa y", zlabel="Predikce")

plot(LL)
plot(log.(LL), label="Log of Loss function")