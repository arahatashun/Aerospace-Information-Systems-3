#!/usr/bin/env julia
# Author:Shun Arahata
# Code for problem 8_3
# Particle Filter for Lorenz Equation
using PyPlot
using Distributions
const σ=10
const ρ=28
const β=8/3
const STEPNUM = 1000000
const OBSERVATION = 100
const STEP = 0.0001
const PARTICLE_NUM = 5

mutable struct ParticleFilter
    weight::Array{Float64,1}
    particle::Array{Float64,3}
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

function predict(filter::ParticleFilter)
    for i in 1:PARTICLE_NUM
        filter.particle[:,i] += runge_kutta(Lorenz,filter.particle[:,i],STEP)
    end
end

function  main()
    x_ini = [0.1; 0.1; 0.1]
    x_true = Array{Float64, 2}(STEPNUM+1,3)
    #initial
    x_true[1,:] = x_ini
    
    for i in 1:STEPNUM
        x_true[i+1,:] = x_true[i,:] + runge_kutta(Lorenz,x_true[i,:],STEP)
    end
    plot_all(x_true)
end

function plot_all(x_true)
    #plot(x_true[1]',x_true[2]',x_true[3]',marker=:circle)
    fig = figure()
    ax = gca(projection="3d")
    ax[:plot](x_true[:,1],x_true[:,2],x_true[:,3])
    PyPlot.plt[:show]()
end

main()