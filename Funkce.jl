using LinearAlgebra
using Plots
using Flux


function init_params!(model, data::Tuple)
    i = data[1]
    j = data[2]
    x = data[3]

    (params(model[i]))[j] .= x

    return nothing
end

function freeze_params!(model, ps, data::Tuple)
    i = data[1]
    j = data[2]

    delete!(ps,params(model[i])[j])

    return nothing
end

function Vypis(ps, gs)
    printstyled("Parametry modelu: \n"; color = :green)
    println(ps) 
    printstyled("Hodnota ztrátové funkce "; color = :green)
    println(LL[end])
    printstyled("|Grad Wr|: "; color = :green)
    println(norm(gs[ps[1]])) 
    printstyled("|Grad Wi|: "; color = :green)
    println(norm(gs[ps[2]])) 
    printstyled("|Grad A|: "; color = :green)
    println(norm(gs[ps[3]])) 
    printstyled("|Grad b|: "; color = :green)
    println(norm(gs[ps[4]]))
    return nothing
end
