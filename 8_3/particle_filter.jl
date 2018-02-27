#!/usr/bin/env julia
# Author:Shun Arahata
# Code for problem 8_3
# Particle Filter for Lorenz Equation
using PyPlot
using Distributions
const σ=10
const ρ=28
const β=8/3
const STEPNUM = 10000
const OBSERVATION = 100
const STEP = 0.0001
const PARTICLE_NUM = 400
const rng = MersenneTwister(8)
const STD = 2.0
const R = [STD 0 0;
            0 STD 0;
            0 0  STD]
function rand_normal(μ, σ)
    #= return a random sample from a normal (Gaussian) distribution
    refering from
    https://www.johndcook.com/blog/2012/02/22/julia-random-number-generation/
    =#
    if σ <= 0.0
        error("standard deviation must be positive")
    end
    u1 = rand(rng)
    u2 = rand(rng)
    r = sqrt( -2.0*log(u1) )
    θ = 2.0*π*u2
    return μ + σ*r*sin(θ)
end

function Lorenz(x::Array{Float64,1})
    #=Lorenz Equation
    \frac{d x}{d t} = \sigma (y-x)\\
    \frac{d y}{d t} = x(\rho-z)-y \\
    \frac{d z}{d t} = xy - \beta z

    :param: xyz
    :return: rhs
    =#
    x_ = x[1]
    y = x[2]
    z = x[3]
    dx = σ * (y - x_)
    dy = x_ * (ρ - z) - y
    dz = x_ * y - β * z 
    return [dx; dy; dz]
end

function runge_kutta(f, x, step)
    #=　Fourth-order Runge-Kutta method.
    :param f: differential equation f(x)
     Note: input output must be the same dimension list
    :param x: variable
    :param step: step time
    :return: increment
    =#
    k1 = f(x)
    k2 = f(x + step / 2 * k1)
    k3 = f(x + step / 2 * k2)
    k4 = f(x + step * k3)
    sum = step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return sum
end

mutable struct ParticleFilter
    weight::Array{Float64,1}
    particle::Array{Float64,2}
end

function initialize_pf()
    particles = Array{Float64, 2}(3,PARTICLE_NUM)
    for i in 1:PARTICLE_NUM
        particles[:,i] = [rand_normal(0.1,STD) for j in 1:3]
    end
    weight = [ 1/PARTICLE_NUM for i in 1:PARTICLE_NUM]
    filter = ParticleFilter(weight, particles)
    return filter
end


function predict!(filter::ParticleFilter)
    for i in 1:PARTICLE_NUM
        filter.particle[:,i] += runge_kutta(Lorenz,filter.particle[:,i],STEP)
    end
end

function getmean(filter::ParticleFilter)
    μ = [0; 0; 0]
    for i in 1:PARTICLE_NUM
        μ += filter.particle[:,i] * filter.weight[i]
    end
    return μ
end

function evaluation!(filter::ParticleFilter,observation::Array{Float64,1})
    for i in 1:PARTICLE_NUM
        x = filter.particle[:,i]
        z = x - observation
        exponent = -1/2 * z' * inv(R) * z
        filter.weight[i] = 1/((2π)^(3/2)*(det(R))^(1/2)) * exp(exponent)
    end
    Σp = sum(filter.weight)
    #println(Σp)
    filter.weight = filter.weight/Σp 
end

function  main()
    x_ini = [0.1; 0.1; 0.1]
    x_true = Array{Float64, 2}(STEPNUM+1,3)
    pf_estimation = Array{Float64, 2}(STEPNUM+1,3)
    simple_estimation = Array{Float64, 2}(STEPNUM+1,3)
    #initial
    x_true[1,:] = x_ini
    pf_estimation[1,:] = x_ini
    simple_estimation[1,:] = x_ini
    filter = initialize_pf()
    for i in 1:STEPNUM
        x_true[i+1,:] = x_true[i,:] + runge_kutta(Lorenz,x_true[i,:],STEP)
        simple_estimation[i+1,:] = simple_estimation[i,:] + runge_kutta(Lorenz,simple_estimation[i,:],STEP)
        predict!(filter)
        pf_estimation[i+1, :] = getmean(filter)
        if i %OBSERVATION == OBSERVATION-1
        observe = x_true[i+1,:] + [rand_normal(0,STD) for j in 1:3]
        evaluation!(filter,observe)
        simple_estimation[i+1,:] = observe
        end
    end
    plot_all(x_true,pf_estimation,simple_estimation)
end

function plot_all(x_true, x_pf,simple)
    time = [STEP*(j-1) for j in 1:STEPNUM+1]
    fig = figure()
    ax = gca(projection="3d")
    ax[:plot](x_true[:,1],x_true[:,2],x_true[:,3])
    ax[:plot](x_pf[:,1],x_pf[:,2],x_pf[:,3])
    ax[:plot](simple[:,1],simple[:,2],simple[:,3])
    PyPlot.plt[:show]()
    fig2 = figure()
    ax =fig2[:add_subplot](2,1,1)
    ax[:plot](time,x_true[:,1],label="true")
    ax[:plot](time,x_pf[:,1],label="particle filter")
    ax[:plot](time,simple[:,1],label="simple")
    legend(loc="right")
    ax =fig2[:add_subplot](2,1,2)
    ax[:plot](time,x_true[:,1]-x_pf[:,1],label="pf error")
    ax[:plot](time,x_true[:,1]-simple[:,1],label="simple error")
    legend(loc="right")
    PyPlot.plt[:show]()
end

main()