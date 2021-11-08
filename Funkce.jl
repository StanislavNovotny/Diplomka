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

#Vypis pro NaiveNPU/Dense Chains
function Vypis1(ps, gs)
    printstyled("Parametry modelu: \n"; color = :yellow)
    println()
    for i = 1:length(model)
        if isequal(typeof(model[i]),NaiveNPU{Matrix{Float32}})
         printstyled("Wr: "; color = :cyan)
         println(ps[2*i-1]) 
         printstyled("Wi: "; color = :cyan)
         println(ps[2*i]) 
        else
         printstyled("A: "; color = :cyan)
         println(ps[2*i-1]) 
         printstyled("b: "; color = :cyan)
         println(ps[2*i])
        end
    end
     println()
    printstyled("Hodnota ztrátové funkce "; color = :blue)
    println(LL[end])
    println()
    for i = 1:length(model)
       if isequal(typeof(model[i]),NaiveNPU{Matrix{Float32}})
        printstyled("|Grad Wr|: "; color = :green)
        println(norm(gs[ps[2*i-1]])) 
        printstyled("|Grad Wi|: "; color = :green)
        println(norm(gs[ps[2*i]])) 
       else
        printstyled("|Grad A|: "; color = :green)
        println(norm(gs[ps[2*i-1]])) 
        printstyled("|Grad b|: "; color = :green)
        println(norm(gs[ps[2*i]]))
       end
    end
    return nothing
end
