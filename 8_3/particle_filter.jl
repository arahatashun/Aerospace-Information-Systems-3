#!/usr/bin/env julia
# Author:Shun Arahata
# Code for problem 8_3
# Particle Filter for Lorenz Equation
using PyPlot
using Distributions
using Calculus

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
const Q = [STD 0 0;
            0 STD 0;
            0 0 STD]

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

mutable struct EKF
    #Extended Kalman Filter
    state::Array{Float64,1}
    variance::Array{Float64,2}
end

function predict!(filter::EKF)
    filter.state += runge_kutta(Lorenz,filter.state,STEP)
    V = filter.variance
    Φ(x,y,z) = expm( [-σ σ 0;ρ-z -1 0 ;y 0 -β]*STEP)
    a11 = derivative(x -> Φ(x,filter.state[2],filter.state[3])[1,1], filter.state[1])
    a12 = derivative(x -> Φ(x,filter.state[2],filter.state[3])[1,2], filter.state[1])
    a13 = derivative(x -> Φ(x,filter.state[2],filter.state[3])[1,3], filter.state[1])
    a21 = derivative(x -> Φ(filter.state[1],x,filter.state[3])[2,1], filter.state[2])
    a22 = derivative(x -> Φ(filter.state[1],x,filter.state[3])[2,2], filter.state[2])
    a23 = derivative(x -> Φ(filter.state[1],x,filter.state[3])[2,3], filter.state[2])
    a31 = derivative(x -> Φ(filter.state[1],filter.state[2],x)[3,1], filter.state[3])
    a32 = derivative(x -> Φ(filter.state[1],filter.state[2],x)[3,2], filter.state[3])
    a33 = derivative(x -> Φ(filter.state[1],filter.state[2],x)[3,3], filter.state[3])
    A = [ a11 a12 a13;a21 a22 a23;a31 a32 a33]
    V = A*V*A'+Q
    filter.variance = V
end

function update(filter::EKF, observation::Array{Float64,1})
    # update and observe
    C = [1 0 0;0 1 0; 0 0 1]
    V = filter.variance
    K = V * C * inv(C * V * C + R)
    filter.state += K * (observation - C *filter.state)
    filter.variance = (eye(3)-K * C) * V
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
    filter.weight = filter.weight/Σp 
end

function  main()
    x_ini = [0.1; 0.1; 0.1]
    x_true = Array{Float64, 2}(STEPNUM+1,3)
    pf_estimation = Array{Float64, 2}(STEPNUM+1,3)
    simple_estimation = Array{Float64, 2}(STEPNUM+1,3)
    ekf_estimation = Array{Float64, 2}(STEPNUM+1,3)
    #initial
    x_true[1,:] = x_ini
    pf_estimation[1,:] = x_ini
    simple_estimation[1,:] = x_ini
    ekf_estimation[1,:] = x_ini
    pf = initialize_pf()
    ekf = EKF(x_ini,R) 
    for i in 1:STEPNUM
        x_true[i+1,:] = x_true[i,:] + runge_kutta(Lorenz,x_true[i,:],STEP)
        simple_estimation[i+1,:] = simple_estimation[i,:] + runge_kutta(Lorenz,simple_estimation[i,:],STEP)
        predict!(pf)
        predict!(ekf)
        pf_estimation[i+1, :] = getmean(pf)
        ekf_estimation[i+1, :] = ekf.state
        if i %OBSERVATION == OBSERVATION-1
        observe = x_true[i+1,:] + [rand_normal(0,STD) for j in 1:3]
        evaluation!(pf,observe)
        update(ekf,observe)
        simple_estimation[i+1,:] = observe
        end
    end
    plot_all(x_true,pf_estimation,simple_estimation,ekf_estimation)
end

function plot_all(x_true, x_pf,simple,ekf)
    time = [STEP*(j-1) for j in 1:STEPNUM+1]
    fig = figure()
    ax = gca(projection="3d")
    ax[:plot](x_true[:,1],x_true[:,2],x_true[:,3],label ="true")
    ax[:plot](x_pf[:,1],x_pf[:,2],x_pf[:,3],label="particle filter")
    ax[:plot](simple[:,1],simple[:,2],simple[:,3],label="simple estimation")
    ax[:plot](ekf[:,1],ekf[:,2],ekf[:,3],label="extended kalman filter")
    legend(loc="right")
    PyPlot.plt[:savefig]("pf.pgf")
    PyPlot.plt[:show]()
    fig2 = figure()
    ax =fig2[:add_subplot](2,1,1)
    ax[:plot](time,x_true[:,1],label="true")
    ax[:plot](time,x_pf[:,1],label="particle filter")
    ax[:plot](time,simple[:,1],label="simple")
    ax[:plot](time,ekf[:,1],label="ekf")
    legend(loc="right")
    ax =fig2[:add_subplot](2,1,2)
    ax[:plot](time,x_true[:,1]-x_pf[:,1],label="pf error")
    ax[:plot](time,x_true[:,1]-simple[:,1],label="simple error")
    ax[:plot](time,ekf[:,1]-simple[:,1],label="ekf error")
    legend(loc="right")
    PyPlot.plt[:show]()
end

main()