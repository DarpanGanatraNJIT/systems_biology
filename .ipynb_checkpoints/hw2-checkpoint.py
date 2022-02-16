#!/usr/bin/env python
# coding: utf-8

# # Darpan Ganatra
# ### Systems Bio
# ### Homework 2
# 
# # Question 1
# 
# ---
# ## 1.a.
# $
# \begin{aligned}
# \dfrac{dS}{dt} &= \Gamma N + \dfrac{-a}{N}(1+\gamma I)IS + \lambda R - \mu S \\
# \dfrac{dI}{dt} &= \dfrac{a}{N}(1+\gamma I)IS - I(c + \phi) \\
# \dfrac{dR}{dt} &= cI - R(\lambda + \mu)
# \end{aligned}
# $
# 
# Where:
# - $\Gamma$ : birth rate
# - $\mu$ : natural death rate
# - $\phi$ : death rate associated with those that die with the infection
# 
# ## 1.b.
# $
# \dfrac{dS}{dt} = \Gamma N + \dfrac{-a}{N}(1+\gamma I)IS + \lambda R - S(\theta + \mu) \\
# \dfrac{dI}{dt} = \dfrac{a}{N}(1+\gamma I)IS + V \bar{\theta} - I(c + \phi) \\
# \dfrac{dR}{dt} = cI - R(\lambda + \mu) \\
# \dfrac{dV}{dt} = S\theta - V\bar{\theta} - \mu V \\
# \dfrac{dN}{dt} = \Gamma N - (S + R + V)\mu - I\phi
# $
# 
# Where:
# - $\Gamma$ : birth rate
# - $\mu$ : natural death rate
# - $\phi$ : death rate associated with those that die with the infection
# - $\theta$ : Vaccination rate
# - $\bar{\theta}$ : Failure of vaccination rate
# 
# 
# Figure attached called "SIR_vax.png"


import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

def sir_model(y, t, N, birth_rate, a, gamma_feedback,lam_feedback, theta, mu, c, theta_bar, phi):
    S, I, R, V = y
    dSdt = birth_rate*N + (-a/N)*(1+gamma_feedback*I)*I*S + lam_feedback*R - S*(theta + mu)
    dIdt = (a/N)*(1+gamma_feedback*I)*I*S + V*theta_bar - I*(c + phi)
    dRdt = c*I - R*(lam_feedback + mu)
    dVdt = S*theta - V*(theta_bar + mu)
    return dSdt, dIdt, dRdt, dVdt

def output(birth_rate, a, gamma_feedback,lam_feedback, theta, mu, c, theta_bar, phi, t_len, N):
    I0, R0, V0 = 1, 0, 0
    S0 = N - I0 - R0 - V0
    t = np.arange(0,t_len)
    y0 = S0, I0, R0, V0
    ret = odeint(sir_model, y0, t, args = (N, birth_rate, a, gamma_feedback,lam_feedback, theta, mu, c, theta_bar, phi))
    S, I, R, V = ret.T
    return S, I, R, V

def plot_output(birth_rate, a, gamma_feedback,lam_feedback, theta, mu, c, theta_bar, phi, t_len, N):
    susceptible, infected, recovered, vaccinated = output(birth_rate = birth_rate, 
                                                          a = a,
                                                          gamma_feedback = gamma_feedback,
                                                          lam_feedback = lam_feedback,
                                                          theta = theta, 
                                                          mu = mu, 
                                                          c = c, 
                                                          theta_bar = theta_bar, 
                                                          phi = phi, 
                                                          t_len = t_len, 
                                                          N = N)
    total_population  = susceptible + infected + recovered + vaccinated
    total_time = np.arange(0, t_len)
    plt.figure(figsize = (14,7))
    plt.plot(total_time, total_population, label = 'N')
    plt.plot(total_time, susceptible, 'b--', label = 'S')
    plt.plot(total_time, infected, 'r', label='I')
    plt.plot(total_time, recovered, 'g--', label='R')
    plt.plot(total_time, vaccinated, 'p--', label='V')
    plt.xlim(0, t_len)
    plt.legend()
    plt.title(f"""a={a}\tc={c}\t $\\theta$={theta}\t $N_{{0}}={N}$""")
    plt.xlabel("t")
    plt.ylabel("Population")
    plt.show()

    
    
plot_output(birth_rate=0.1, a= 0.5, gamma_feedback=0.5, lam_feedback=0.5, theta=0.1, mu = 0.1, c = 0.1, theta_bar=0.01, phi = 0.1, t_len = 100, N=100)

# # Question 2
# 
# In order to have interacting populations, I need to figure out the parameters which will change as well as the populations which will be allowed to travel: 
# 
# Parameters which may change:
# - $S, I, R, V, N$
# - $\theta$
# 
# The populations which are allowed to travel are:
# - Those that already have recovered ($R$, assumed to have developed immunity)
# - Those that are vaccinated ($V$)
# 
# So the model should be: 
# 
# Population 1
# ---
# 
# $
# \dfrac{dS_{1}}{dt} = \Gamma N_{1} + \dfrac{-a}{N_{1}}(1+\gamma I_{1})I_{1}S_{1} + \lambda R_{1} - S_{1}(\theta_{1} + \mu) \\
# \dfrac{dI_{1}}{dt} = \dfrac{a}{N_{1}}(1+\gamma I_{1})I_{1}S_{1} + V_{1} \bar{\theta} - I_{1}(c + \phi) \\
# \dfrac{dR_{1}}{dt} = cI_{1} - R_{1}(\lambda + \mu +\beta) + \bar{\beta}R_{2} \\
# \dfrac{dV_{1}}{dt} = S_{1}\theta_{1} - V_{1}(\bar{\theta} + \mu +\beta) + \bar{\beta}V_{2} \\
# \dfrac{dN_{1}}{dt} = \Gamma N_{1} - (S_{1} + R_{1} + V_{1})\mu - I_{1}\phi + \bar{\beta}(R_{2} + V_{2}) - \beta (R_{1} + V_{1})
# $
# 
# Population 2
# ---
# 
# $
# \dfrac{dS_{2}}{dt} = \Gamma N_{2} + \dfrac{-a}{N_{2}}(1+\gamma I_{2})I_{2}S_{2} + \lambda R_{2} - S_{2}(\theta_{2} + \mu) \\
# \dfrac{dI_{2}}{dt} = \dfrac{a}{N_{2}}(1+\gamma I_{2})I_{2}S_{2} + V_{2} \bar{\theta} - I_{2}(c + \phi) \\
# \dfrac{dR_{2}}{dt} = cI_{2} - R_{2}(\lambda + \mu + \bar{\beta}) + \beta R_{1} \\
# \dfrac{dV_{2}}{dt} = S_{2}\theta_{2} - V_{2}(\bar{\theta} + \mu +\bar{\beta}) + \beta V_{1} \\
# \dfrac{dN_{2}}{dt} = \Gamma N_{2} - (S_{2} + R_{2} + V_{2})\mu - I_{2}\phi + \beta (R_{1} + V_{1}) - \bar{\beta} (R_{2} + V_{2})
# $
# 
# Figure attached called "TwoPopulation.png"



def dual_population_model(y, t, N1, N2, birth_rate, a, gamma_feedback, lam_feedback, theta1, theta2, mu, c, theta_bar, phi, beta, beta_bar):
    S1,S2,I1,I2,R1,R2,V1,V2 = y
    dSdt1 = birth_rate*N1 + (-a/N1)*(1 + gamma_feedback*I1)*I1*S1 + lam_feedback*R1 - S1*(theta1 + mu)
    dSdt2 = birth_rate*N2 + (-a/N2)*(1 + gamma_feedback*I2)*I2*S2 + lam_feedback*R2 - S2*(theta2 + mu)
    
    dIdt1 = (a/N1)*(1 + gamma_feedback*I1)*I1*S1 + V1*theta_bar - I1*(c + phi)
    dIdt2 = (a/N2)*(1 + gamma_feedback*I2)*I2*S2 + V2*theta_bar - I2*(c + phi)
    
    dRdt1 = c*I1 - R1*(lam_feedback + mu + beta) + beta_bar*R2
    dRdt2 = c*I2 - R2*(lam_feedback + mu + beta_bar) + beta*R1
    
    dVdt1 = S1*theta1 - V1*(theta_bar + mu + beta) + beta_bar*V2
    dVdt2 = S2*theta2 - V2*(theta_bar + mu + beta_bar) + beta*V1
    return dSdt1, dSdt2, dIdt1, dIdt2, dRdt1, dRdt2, dVdt1, dVdt2

def output(birth_rate, a, gamma_feedback, lam_feedback, theta1, theta2, mu, c, theta_bar, phi, beta, beta_bar, t_len, N1, N2):
    I10, I20, R10, R20, V10, V20 = 1, 1, 0, 0, 0, 0
    S10 = N1 - I10 - R10 - V10
    S20 = N2 - I20 - R20 - V20
    t = np.arange(0, t_len)
    y0 = S10, S20, I10, I20, R10, R20, V10, V20
    ret = odeint(dual_population_model, y0, t, args = (N1, N2, birth_rate, a, gamma_feedback, lam_feedback, theta1, theta2, mu, c, theta_bar, phi, beta, beta_bar))
    S1, S2, I1, I2, R1, R2, V1, V2 = ret.T
    return S1, S2, I1, I2, R1, R2, V1, V2

def plot_output(birth_rate, a, gamma_feedback, lam_feedback, theta1, theta2, mu, c, theta_bar, phi, beta, beta_bar, t_len, N1, N2):
    sus1, sus2, inf1, inf2, rec1, rec2, vac1, vac2 = output(birth_rate = birth_rate,
                                                            a = a,
                                                            gamma_feedback = gamma_feedback,
                                                            lam_feedback = lam_feedback,
                                                            theta1 = theta1,
                                                            theta2 = theta2,
                                                            mu = mu,
                                                            c = c,
                                                            theta_bar = theta_bar,
                                                            phi = phi,
                                                            beta = beta,
                                                            beta_bar = beta_bar,
                                                            t_len = t_len,
                                                            N1 = N1,
                                                            N2 = N2)
    N1 = sus1 + inf1 + rec1 + vac1
    N2 = sus2 + inf2 + rec2 + vac2
    total_time = np.arange(0, t_len)
    plt.figure(figsize = (14,7))
    plt.plot(total_time, N1, label = "City 1")
    plt.plot(total_time, N2, label = "City 2")
    plt.title(f"City 1 -> City 2: {beta}\nCity 2 -> City 1: {beta_bar}")
    plt.xlabel("t")
    plt.ylabel("Population")
    plt.legend()
    plt.show()

    
plot_output(birth_rate=0.2, a = 0.5, gamma_feedback=0.5, lam_feedback=0.5, theta1=0.7, theta2=0.1, mu = 0.1, c = 0.2, theta_bar=0.01, phi = 0.15, beta=0.3, beta_bar = 0.2, t_len=100, N1 = 500, N2 = 500)


# # Question 3
# 
# There was a question from Camilo about what would happen if there was the possibility of lower birth rates as a result of a particular infection rate (In this example it's relatively high at 50%, just so I can see the dynamic). I was also interested to see the dynamic affecting a larger city vs. a small town.




def dual_population_model(t, y, N1, N2, birth_rate, a, gamma_feedback, lam_feedback, theta1, theta2, mu, c, theta_bar, phi, beta, beta_bar):
    S1,S2,I1,I2,R1,R2,V1,V2 = y
    norm_birth_rate = birth_rate
    if I1/N1 >= 0.5 or I2/N2 >=0.5:
        birth_rate = max(0.001, norm_birth_rate/2)
    elif I1/N1 >= 0.5 and I2/N2 >=0.5:
        birth_rate = 0
    else:
        birth_rate = norm_birth_rate
    
    dSdt1 = birth_rate*N1 + (-a/N1)*(1 + gamma_feedback*I1)*I1*S1 + lam_feedback*R1 - S1*(theta1 + mu)
    dSdt2 = birth_rate*N2 + (-a/N2)*(1 + gamma_feedback*I2)*I2*S2 + lam_feedback*R2 - S2*(theta2 + mu)
    
    dIdt1 = (a/N1)*(1 + gamma_feedback*I1)*I1*S1 + V1*theta_bar - I1*(c + phi)
    dIdt2 = (a/N2)*(1 + gamma_feedback*I2)*I2*S2 + V2*theta_bar - I2*(c + phi)
    
    dRdt1 = c*I1 - R1*(lam_feedback + mu + beta) + beta_bar*R2
    dRdt2 = c*I2 - R2*(lam_feedback + mu + beta_bar) + beta*R1
    
    dVdt1 = S1*theta1 - V1*(theta_bar + mu + beta) + beta_bar*V2
    dVdt2 = S2*theta2 - V2*(theta_bar + mu + beta_bar) + beta*V1
    return dSdt1, dSdt2, dIdt1, dIdt2, dRdt1, dRdt2, dVdt1, dVdt2

def output(birth_rate, a, gamma_feedback, lam_feedback, theta1, theta2, mu, c, theta_bar, phi, beta, beta_bar, t_len, N1, N2):
    I10, I20, R10, R20, V10, V20 = 1, 1, 0, 0, 0, 0
    S10 = N1 - I10 - R10 - V10
    S20 = N2 - I20 - R20 - V20
    t = np.arange(0, t_len)
    y0 = S10, S20, I10, I20, R10, R20, V10, V20
    ret = solve_ivp(dual_population_model, [0, t_len], y0, args = (N1, N2, birth_rate, a, gamma_feedback, lam_feedback, theta1, theta2, mu, c, theta_bar, phi, beta, beta_bar))
    return ret


def plot_output(birth_rate, a, gamma_feedback, lam_feedback, theta1, theta2, mu, c, theta_bar, phi, beta, beta_bar, t_len, N1, N2):
    ret_vals = output(birth_rate = birth_rate,
                        a = a,
                        gamma_feedback = gamma_feedback,
                        lam_feedback = lam_feedback,
                        theta1 = theta1,
                        theta2 = theta2,
                        mu = mu,
                        c = c,
                        theta_bar = theta_bar,
                        phi = phi,
                        beta = beta,
                        beta_bar = beta_bar,
                        t_len = t_len,
                        N1 = N1,
                        N2 = N2)
    sus1, sus2, inf1, inf2, rec1, rec2, vac1, vac2 = ret_vals.y
    total_time = ret_vals.t
    N1 = sus1 + inf1 + rec1 + vac1
    N2 = sus2 + inf2 + rec2 + vac2
    plt.figure(figsize = (14,7))
    plt.plot(total_time, N1, label = "City 1")
    plt.plot(total_time, N2, label = "City 2")
    plt.title(f"City 1 -> City 2: {beta}\nCity 2 -> City 1: {beta_bar}")
    plt.xlabel("t")
    plt.xlim(0, t_len)
    plt.ylim(bottom = 0)
    plt.ylabel("Population")
    plt.legend()
    plt.show()
    
    plt.figure(figsize = (14,7))
    plt.plot(total_time, inf1/N1, label = 'Infection Ratio 1')
    plt.plot(total_time, inf2/N2, label = 'Infection Ratio 2')
    plt.axhline(0.7, 0, color = 'k')
    plt.legend()
    plt.xlabel("Infection Ratio")
    plt.ylabel("Time")
    plt.show()
    print(f"Max Number of Infected in City 1:\n {round(max(inf1))}\nMax Number of infected in City 2:\n {round(max(inf2))}" )
    
plot_output(birth_rate=0.1, a = 0.5, gamma_feedback=0.5, lam_feedback=0.5, theta1=0.7, theta2=0.2, mu = 0.1, c = 0.2, theta_bar=0.01, phi = 0.1, beta=0.01, beta_bar = 0.1, t_len=100, N1 = 5000, N2 = 500)

