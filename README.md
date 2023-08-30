# Sistemas de comunicación inalámbrica MIMO asistidos por RIS con modulación QAM/QSM
Códigos en Python de simulaciones MIMO asistidas por superficies inteligentes reconfigurables (RIS). Los bits son modulados por medio de esquemas QAM y EQSM.
Los códigos al ejecutarse entregan las curvas de Probabilidad de Error de Bit promedio (ABEP) vs Relación Señal a Ruído.
En el caso del algoritmo de Modulación Espacial Extendida por Cuadratura (EQSM), el algoritmo se implementó con base en lo descrito por [[1]](#1).
Las librerías necesarias para ejecutar el código de las simulaciones son las siguientes:
* Numpy
* Matplotlib
* Scipy
Para instalarlas de manera automática, se puede ejecutar el comando  `pip install -r requirements.txt`
## Descripción del sistema
La estación base comprende $N_t$ antenas transmisoras, $K$ usuarios o estaciones móviles con $N_r$ antenas receptoras. 
Cada sistema usa $N$ superficies RIS para reflejar la señal y cada RIS emplea $N_s$ espejos reflectores.
En el caso de los canales inalámbricos de transmisión, se definen los siguientes:

* La matriz de canal enre la estación base y el $k$-ésimo usuario de los $K$ disponibles ($k \in K$):
$$\mathbf{H}_{n,k}^{DP} \in \mathbb{C}^{Nt \times N_r}$$
* La matriz de canal entre la $n$-ésima superficie RIS de las $N$ disponibles ($n \in N$) y el $k$-ésimo usuario:
$$\mathbf{H_n,k} \in \mathbb{C}^{N_r \times N_s}$$
* La matriz de canal entre la estación base y la $n$-ésima superficie RIS:
$$G_n \in \mathbb{C}^{N_s \times N_t}$$

El modelo de canal se define como cuasi-estático, el desvanecimiento correlacionado con una distribución Rayleigh, donde suponemos que sus elementos son variables aleatorias gaussianas complejas con media cero y varianza unitaria $\mathcal{C}\mathcal{N}(0,1)$.


## Referencias
<a id="1">[1]</a> 
Castillo-Soria, F. R., Cortez, J., Gutiérrez, C. A., Luna-Rivera, M., & Garcia-Barrientos, A. (2019). 
Extended quadrature spatial modulation for MIMO wireless communications. 
Physical Communication, 32, 88-95.
