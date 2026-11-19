"""
Práctica 4: Modelado y control del sistema endocrino

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Leonardo Zaid Moreno Zavala
Número de control: 24210590
Correo institucional: l24210590@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
import matplotlib.pyplot as plt
import numpy as np
import control as ctrl

x0, t0, tend, dt, w, h = 0, 0, 10, 1E-4, 7.5, 4.6
n = round((tend - t0) / dt) + 1
t = np.linspace(t0, tend, n)

Ve = np.ones(n)

kP = 1298.6989
kI = 312346.1725
kD = 0.17968      
N  = 4964.0941    

Cr = 1E-6
Re = 1/(Cr*kI)
Rr = kP*Re
Ce = kD/Rr
numPID = [Re*Rr*Ce,Re*Ce+Rr*Cr,1]
denPID = [Re*Cr,0]
PID = ctrl.tf(numPID,denPID)
print(f"Capacitancia Cr: {Cr} Faradios\n")
print(f"Resistencia Re: {Re} Ohms\n")
print(f"Resistencia Rr: {Rr} Ohmsd\n")
print(f"Capacitancia Ce: {Ce} Faradios\n")
print(f"Funcion de transferencia del controlador: {PID}\n")

R1, L = 1e3, 0.1

def G(C, R2):
    return ctrl.tf([L, R2], [R1 * L * C, L + R1 * R2 * C, R1 + R2])

control = G(1e-6, 100e3)      
caso    = G(1e-3, 1e3)        
print(f"Funcion de transferencia del control: {control}\n")
print(f"Funcion de transferencia del caso: {caso}\n")
print("Polos del control:", control.poles())
print("Polos del caso   :", caso.poles())

s = ctrl.tf("s")
PID = kP + kI / s if kD == 0 else kP + kI / s + kD * N * s / (s + N)

sysPID = ctrl.feedback(ctrl.series(PID, caso), 1, sign=-1)
print(f"\nFuncion de transferencia en lazo cerrado: {sysPID}\n")
print("Polos del lazo cerrado:", sysPID.poles())
print("Lazo cerrado estable:", bool(np.all(sysPID.poles().real < 0)))

def closeloop(t, sys, u):
    _, y = ctrl.forced_response(sys, t, u, x0)
    return y

_, Vsx = ctrl.forced_response(control, t, Ve)     
_, Vsy = ctrl.forced_response(caso, t, Ve)        
tratamiento = closeloop(t, sysPID, Vsx)           

print(f"\nVsx max (control) = {Vsx.max():.4f} V")
print(f"Vsy max (caso)    = {Vsy.max():.4f} V")
print(f"PID max / min     = {tratamiento.max():.4f} / {tratamiento.min():.4f} V")


def plotsignals(t, Ve, PID, Vsy, Vsx, nombre="Caso", w=7.5, h=4.6, ylim=(-0.2, 1.2)):
    plt.rcParams.update({"font.family": "STIXGeneral",
                         "mathtext.fontset": "stix",
                         "font.size": 12})
    clr1 = np.array([90, 90, 90]) / 255       # Ve: gris
    clr2 = np.array([0, 158, 115]) / 255      # PID: verde
    clr3 = np.array([213, 94, 0]) / 255       # Caso: naranja
    clr4 = np.array([0, 114, 178]) / 255      # Control: azul

    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(t, Ve,  "-",  color=clr1, lw=1.2, label=r"$V_e(t)$", clip_on=False)
    ax.plot(t, PID, "--", color=clr2, lw=1.4, label=r"$PID(t)$ : Tratamiento", clip_on=False)
    ax.plot(t, Vsy, ":",  color=clr3, lw=1.8, label=r"$V_{sY}(t)$ : Caso", clip_on=False)
    ax.plot(t, Vsx, "-.", color=clr4, lw=1.4, label=r"$V_{sX}(t)$ : Control", clip_on=False)

    ax.set_xlim(-0.2, 10);  ax.set_xticks(np.arange(0, 11, 1))
    ax.set_ylim(*ylim);     ax.set_yticks(np.arange(ylim[0], ylim[1] + 1e-9, 0.2))
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$V_e(t)$ [V]")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=4,
              frameon=True, fancybox=False, edgecolor="k",
              prop={"style": "italic", "size": 10},
              handlelength=2.2, columnspacing=1.0)
    fig.savefig(f"{nombre}.pdf", bbox_inches="tight")
    plt.show()
    fig.savefig('CasoTratamiento_python.pdf',bbox_inches='tight')

plotsignals(t, Ve, tratamiento, Vsy, Vsx, "Caso", w, h)