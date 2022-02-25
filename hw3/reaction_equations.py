#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def enzymatic_reactions(t, y, k_sc, k_cs, k_cp):
    S, C, E, P = y
    dS = -k_sc * S * E + k_cs * C
    dC = k_sc * S * E - C * (k_cs + k_cp)
    dE = -k_sc * S * E + (k_cs + k_cp) * C
    dP = k_cp * C
    return dS, dC, dE, dP


def output(k_sc, k_cs, k_cp, E0, t_len):
    S0, C0, P0 = 5, 0, 0
    y0 = S0, C0, E0, P0
    ret = solve_ivp(enzymatic_reactions, [0, t_len], y0, args=(k_sc, k_cs, k_cp))
    return ret


def plot_output(k_sc, k_cs, k_cp, E0, t_len):
    ret_vals = output(k_sc, k_cs, k_cp, E0, t_len)
    S, C, E, P = ret_vals.y
    plt.figure(figsize=(14, 7))
    plt.plot(ret_vals.t, S, label="S")
    plt.plot(ret_vals.t, C, "--", label="C")
    plt.plot(ret_vals.t, E, label="E")
    plt.plot(ret_vals.t, P, label="P")
    plt.legend()
    plt.ylabel("Concentration")
    plt.xlabel("Time")
    plt.show()


k1 = 30
k1_neg = 1
k_2 = 10
E_init = 1
t_max = 1


plot_output(k_sc=k1, k_cs=k1_neg, k_cp=k_2, E0=E_init, t_len=t_max)
