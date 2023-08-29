import numpy as np
import matplotlib.pyplot as plt
from comm_utilities import *
from scipy.linalg import svd

Nr = 2
Nt = 8
Nrel = 32
SNR_dB = np.arange(start=-10, stop=32, step=2)
SNR_l = 10**(SNR_dB/10)
BER = np.empty([len(SNR_dB),])
n=0
P_prom_real = 0
M=4
K=2
qam_sym = qam_symbol_generator(M)
SS = np.array(eqsm_constellation(M,qam_sym,Nr,K))
tam_SS = len(SS)
bcpu = int(np.log2(tam_SS))
norma = np.abs(SS)
norma2 = norma[:,0] + norma[:,1]
pot = sum(norma2)/tam_SS
SS /= pot
pot2 =  sum(norma2)/tam_SS
P = 1
At = np.sqrt(P)
n1 = 0
n_iter = 10**4
be = 0
for j, snr in enumerate(SNR_l):
    for u in range(n_iter):
        dato = np.random.randint(256)
        dato2 = np.random.randint(256)
        dato3 = np.random.randint(256)
        dato4 = np.random.randint(256)
        x1 = SS[dato,:]
        x2 = SS[dato2,:]
        x3 = SS[dato3,:]
        x4 = SS[dato4,:]
        x_vec = np.hstack((x1.T,x2.T,x3.T,x4.T)).reshape(8,1)
        norma3 = np.abs(x_vec)
        P_real = sum(norma3)
        P_prom_real += P_real
        n += 1
        H_SR = 1
        H12 = H_channel(2,Nrel)
        H22 = H_channel(2,Nrel)
        H32 = H_channel(2,Nrel)
        H42 = H_channel(2,Nrel)
        H_RD = np.vstack((H12,H22,H32,H42))
        Heq = H_RD*H_SR
        H1 = H12*H_SR
        H2 = H22*H_SR
        H3 = H32*H_SR
        H4 = H42*H_SR
        Hc1 = np.vstack((H2,H3,H4))
        Hc2 = np.vstack((H1,H3,H4))
        Hc3 = np.vstack((H1,H2,H4))
        Hc4 = np.vstack((H1,H2,H3))
        [U1,S1,V1] = svd(Hc1)
        [U2,S2,V2] = svd(Hc2)
        [U3,S3,V3] = svd(Hc3)
        [U4,S4,V4] = svd(Hc4)
        W1 = V1.T.conj()[:,6:8]
        W2 = V2.T.conj()[:,6:8]
        W3 = V3.T.conj()[:,6:8]
        W4 = V4.T.conj()[:,6:8]
        Tx = np.matmul(W1,x_vec[0:2]) + np.matmul(W2,x_vec[2:4]) + np.matmul(W3,x_vec[4:6]) + np.matmul(W4,x_vec[6:8])
        Tx *= P*np.sqrt(snr)
        Rx1 = (1/At)*(H_SR*Tx)+n1
        Tx2 = Rx1
        n2 = awgn_noise(1,Nr)
        y2 = (1/At)*np.matmul(H12,Tx2) + n2
        C2 = np.matmul(H1,W1)
        C = np.matmul(SS,C2.T)
        p = np.sqrt(snr)
        s1 = np.abs(y2[0]-p*C[:,0])**2
        s2 = np.abs(y2[1]-p*C[:,1])**2
        s = s1 + s2
        index = np.argmin(s)
        if index != dato:
            a = biterr_calculation(dato,index,bcpu)
            be += a
    BER[j] = be/(n_iter*bcpu)
    be = 0
    P_prom_real /= n
    n = 0
    print(f'BER = {BER[j]}')
    print(f'P_prom_real = {P_prom_real}')
fig,ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(SNR_dB,BER,color='r',marker='o',linestyle='-')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('ABEP')
ax.set_title('ABEP RIS 4 users 8 bcpu QSM modulation')
plt.grid()
plt.show()





