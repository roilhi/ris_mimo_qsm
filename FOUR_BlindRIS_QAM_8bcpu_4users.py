import numpy as np
import matplotlib.pyplot as plt
from comm_utilities import qam_symbol_generator, qam_mimo_tx_combinations, biterr_calculation, H_channel, awgn_noise
from scipy.linalg import svd

Nr = 2 # Rx antennas/user
Nt = 8 # NTx base station
N_rel = 32 # Number of RIS elements
SNR_dB = np.arange(start=-10, stop=32, step=2)
SNR_l = 10**(SNR_dB/10)
be = 0
BER = np.empty([len(SNR_dB),])
# QAM constellation
M = 16 # constellation order
qam_sym = np.array(qam_symbol_generator(M))
FN = 1/np.sqrt((2/3)*(M-1))
qam_sym *= FN 
suma = 0
for a in range(M):
    pot1 = np.sqrt(np.real(qam_sym[a])**2+np.imag(qam_sym[a])**2)
    suma += pot1
pot = suma/M 
qam_sym /= pot 
qam_sym /= 2

SS = qam_mimo_tx_combinations(Nr,qam_sym)
tam_SS = int(len(SS))
norma = np.abs(SS)
norma2 = norma[:,0] + norma[:,1]
pot = sum(norma2)/tam_SS
SS /= pot

n = 0
P_prom_real = 0
N_iter =  10**4
P = 1
At = np.sqrt(P)
bcpu = int(np.log2(tam_SS))
for j, snr in enumerate(SNR_l):
    P_acu = 0
    for u in range(N_iter):
        dato = np.random.randint(tam_SS)
        dato2 = np.random.randint(tam_SS)
        dato3 = np.random.randint(tam_SS)
        dato4 = np.random.randint(tam_SS)
        x1 = SS[dato,:]
        x2 = SS[dato2,:]
        x3 = SS[dato3,:]
        x4 = SS[dato4,:]
        x_vec = np.hstack((x1.T,x2.T,x3.T,x4.T)).reshape(8,1)
        norma3 = np.abs(x_vec)
        P_real = np.sum(norma3)
        n += 1
        P_prom_real += P_real
        # H_channel
        H_SR1 = H_channel(N_rel,Nt)
        H_SR2 = H_channel(N_rel,Nt)
        H_SR3 = H_channel(N_rel,Nt)
        H_SR4 = H_channel(N_rel,Nt)

        H11 = H_channel(2,N_rel)
        H21 = H_channel(2,N_rel)
        H31 = H_channel(2,N_rel)
        H41 = H_channel(2,N_rel)

        H12 = H_channel(2,N_rel)
        H22 = H_channel(2,N_rel)
        H32 = H_channel(2,N_rel)
        H42 = H_channel(2,N_rel)

        H13 = H_channel(2,N_rel)
        H23 = H_channel(2,N_rel)
        H33 = H_channel(2,N_rel)
        H43 = H_channel(2,N_rel)

        H14 = H_channel(2,N_rel)
        H24 = H_channel(2,N_rel)
        H34 = H_channel(2,N_rel)
        H44 = H_channel(2,N_rel)

        # Direct path
        H_SD1 = H_channel(2,Nt)
        H_SD2 = H_channel(2,Nt)
        H_SD3 = H_channel(2,Nt)
        H_SD4 = H_channel(2,Nt)
        # Composed RIS channel 1 to all users
        H_RD1 = np.vstack((H11,H21,H31,H41))

        H_RD2 = np.vstack((H12,H22,H32,H42))

        H_RD3 = np.vstack((H13,H23,H33,H43))

        H_RD4 = np.vstack((H14,H24,H34,H44))

        # Direct path composed matrix for all users
        H_SD = np.vstack((H_SD1,H_SD2,H_SD3,H_SD4))
        # Equivalent matrix of the system
        Heq = H_RD1 @ H_SR1 + H_RD2 @ H_SR2 + H_RD3 @ H_SR3 + H_RD4 @ H_SR4 + H_SD
        # Another way to calculate the equivalent channels
        H1 = Heq[0:2,:]
        H2 = Heq[2:4,:]
        H3 = Heq[4:6,:]
        H4 = Heq[6:8,:]
        #H1 = (H11 @ H_SR1) + (H12 @ H_SR2) + H_SD1
        #H2 = (H21 @ H_SR1) + (H22 @ H_SR2) + H_SD2
        #H3 = (H31 @ H_SR1) + (H32 @ H_SR2) + H_SD3
        #H4 = (H41 @ H_SR1) + (H42 @ H_SR2) + H_SD4
        # Auxiliar matrices for BD
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
        Tx = W1 @ x_vec[0:2] + W2 @ x_vec[2:4] + W3 @ x_vec[4:6] + W4 @ x_vec[6:8]
        P_acu += np.sum(np.abs(Tx))               
        Tx *= P*np.sqrt(snr)
        # RIS reception
        Rx1 = (1/At) * (H_SR1 @ Tx)
        Rx2 = (1/At) * (H_SR2 @ Tx)
        Rx3 = (1/At) * (H_SR3 @ Tx)
        Rx4 = (1/At) * (H_SR4 @ Tx)

        # Transmit data from RIS
        Tx2 = Rx1
        Tx3 = Rx2
        Tx4 = Rx3
        Tx5 = Rx4
        noise = awgn_noise(1,Nr)
        # Rx from user 1
        y2 = (1/At)*(H11 @ Tx2) + (1/At)*(H12 @ Tx3) + (1/At)*(H13 @ Tx4) + (1/At)*(H14 @ Tx5) + H_SD1 @ Tx + noise
        C2 = H1 @ W1
        C = SS @ C2.T
        # ML detector
        s1 = np.square(np.abs(y2[0]-np.sqrt(snr)*C[:,0]))
        s2 = np.square(np.abs(y2[1]-np.sqrt(snr)*C[:,1]))
        # MRC combiner
        s = s1 + s2
        index = np.argmin(s)
        if index != dato:
            a = biterr_calculation(dato,index,bcpu)
            be += a
    BER[j] = be/(N_iter*bcpu)
    be = 0
    P_prom_real /= n
    n = 0
    print(f'P_prom_real = {P_prom_real} ')
    print(f'BER = {BER[j]}')

fig,ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(SNR_dB,BER,color='r',marker='o',linestyle='-')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('ABEP')
ax.set_title('ABEP 4 RIS 4 users 8 bcpu QAM modulation')
plt.grid()
plt.show()
