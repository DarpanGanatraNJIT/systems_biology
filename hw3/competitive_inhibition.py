#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def competitive_inhibition(t, y, k_sc, k_cs, k_cp, k_id, k_di):
    S, C, E, P, D, I = y
    dS = -k_sc * S * E + k_cs * C
    dC = k_sc * S * E - C * (k_cs + k_cp)
    dE = -k_sc * S * E + (k_cs + k_cp) * C - k_id * I * E + k_di * D
    dP = k_cp * C
    dD = k_id * I * E - k_di * D
    dI = -k_id * I * E + k_di * D
    return dS, dC, dE, dP, dD, dI


def output(k_sc, k_cs, k_cp, k_id, k_di, init_vals, t_len):
    """
    init_vals = S0, C0, E0, P0, D0, I0
    """
    y0 = init_vals
    ret = solve_ivp(
        fun=competitive_inhibition,
        t_span=[0, t_len],
        y0=y0,
        args=(k_sc, k_cs, k_cp, k_id, k_di),
    )
    return ret


def plot_output(
    k_sc, k_cs, k_cp, k_id, k_di, S0, C0, E0, P0, D0, I0, t_len, plot_title
):
    init_vals = S0, C0, E0, P0, D0, I0
    ret_vals = output(k_sc, k_cs, k_cp, k_id, k_di, init_vals, t_len)
    S, C, E, P, D, I = ret_vals.y
    plt.figure(figsize=(14, 7))
    plt.title(plot_title)
    plt.xlim(left=0, right=t_len)
    # plt.ylim(bottom = 0, top = 2)
    plt.plot(ret_vals.t, S, label="S")
    plt.plot(ret_vals.t, C, "--", label="C")
    plt.plot(ret_vals.t, E, label="E")
    plt.plot(ret_vals.t, P, label="P")
    plt.plot(ret_vals.t, D, label="D")
    plt.plot(ret_vals.t, I, label="I")
    # plt.plot(ret_vals.t, C + E, 'k--', label = 'Total Enzyme Constant ($e_{T} = e + c$)')
    plt.legend()
    plt.ylabel("Concentration")
    plt.xlabel("Time")
    plt.show()


plot_output(
    k_sc=30,
    k_cs=1,
    k_cp=10,
    k_id=1,
    k_di=1,
    S0=1,
    C0=0,
    E0=1,
    P0=0,
    D0=0,
    I0=0,
    t_len=1,
    plot_title="Test",
)
