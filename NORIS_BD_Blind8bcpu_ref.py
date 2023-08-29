import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from comm_utilities import *
from numpy.random import standard_normal

SNR_dB = np.arange(start=-10, stop=32, step=2)
SNR_l = 10**(SNR_dB/10)
be = 0
pot = 0
Nt = 8
Nr = 2
BER = []
M = 16
n_iter = 10**4
#i = np.arange(M)
#f = QAMModem(M)
#QAM_sym = f.modulate(dec2bitarray(i,4))
#idx_bin = [2,3,1,0,6,7,5,4,14,15,13,12,10,11,9,8]
#QAM_sym = np.array([QAM_sym[idM] for idM in idx_bin])
QAM_sym = np.array(qam_symbol_generator(M))
FN = 1/np.sqrt((2/3)*(M-1))
QAM_sym = FN*QAM_sym
#SS = np.array([0,0])
#for A1 in range(M):
#    for A2 in range(M):
#        SS = np.vstack((SS,np.array([QAM_sym[A1],QAM_sym[A2]])))

#SS = np.delete(SS,(0), axis=0)
SS = qam_mimo_tx_combinations(Nr,QAM_sym)
tam_SS = int(len(SS))
bcpu = int(np.log2(tam_SS))
norma = np.abs(SS)
norma2 = norma[:,0] + norma[:,1]
pot = sum(norma2)/tam_SS
SS /= pot
B = SS
ni = 0
P_prom_real = 0
for snr in SNR_l:
    for u in range(n_iter):
        dato = np.random.randint(tam_SS)
        dato2 = np.random.randint(tam_SS)
        dato3 = np.random.randint(tam_SS)
        dato4 = np.random.randint(tam_SS)
        x1 = B[dato,:]
        x2 = B[dato2,:]
        x3 = B[dato3,:]
        x4 = B[dato4,:]
        x_vec = np.hstack((x1.T,x2.T,x3.T,x4.T)).reshape(8,1)
        norma3 = np.abs(x_vec)
        P_real = sum(norma3)
        ni+=1
        P_prom_real += P_real
        # Canal H
        H1 = (1/np.sqrt(2))*(standard_normal((2,Nt))+1j*(standard_normal((2,Nt))))
        H2 = (1/np.sqrt(2))*(standard_normal((2,Nt))+1j*(standard_normal((2,Nt))))
        H3 = (1/np.sqrt(2))*(standard_normal((2,Nt))+1j*(standard_normal((2,Nt))))
        H4 = (1/np.sqrt(2))*(standard_normal((2,Nt))+1j*(standard_normal((2,Nt))))

        Hc1 = np.vstack((H2, H3, H4))
        Hc2 = np.vstack((H1, H3, H4))
        Hc3 = np.vstack((H1, H2, H4))
        Hc4 = np.vstack((H1, H2, H3))
        [U1,S1,V1] = svd(Hc1)
        [U2,S2,V2] = svd(Hc2)
        [U3,S3,V3] = svd(Hc3)
        [U4,S4,V4] = svd(Hc4)
        W1 = V1.T.conj()[:,6:8]
        W2 = V2.T.conj()[:,6:8]
        W3 = V3.T.conj()[:,6:8]
        W4 = V4.T.conj()[:,6:8]
        Tx = np.matmul(W1,x_vec[0:2]) + np.matmul(W2,x_vec[2:4]) + np.matmul(W3,x_vec[4:6]) + np.matmul(W4,x_vec[6:8])
        n1  = (1/np.sqrt(2))*standard_normal((2,1))+(1/np.sqrt(2))*standard_normal((2,1))*1j
        # Receptor
        Rx1 = np.sqrt(snr)*(np.matmul(H1,Tx))+n1
        # Detector ML símbolo QAM
        C1 = np.matmul(H1,W1)
        D = np.matmul(C1,B.T)
        A = np.sqrt(snr)
        s1 = np.abs(Rx1[0,0]-A*D[0,:])**2
        s2 = np.abs(Rx1[1,0]-A*D[1,:])**2
        s = s1+s2
        index = np.argmin(s)
        if index != dato:
            #dato_bit = dec2bitarray(dato,8)
            #index_bit = dec2bitarray(index,8)
            #a = len(np.argwhere(dato_bit != index_bit))
            a = biterr_calculation(dato,index,bcpu)
            be += a
    BER.append(be/(n_iter*bcpu))
    P_prom_real /= ni
    be = 0
    ni = 0
    print(f'Potencia promedio real {P_prom_real}')
    print(f' BER = {BER}\n')
fig,ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(SNR_dB,BER,color='b',marker='o',linestyle='-')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('ABEP')
ax.set_title('Blind RIS 8bcpu using QAM')
plt.grid()
plt.show()


'''

P_prom_real=P_prom_real/(ni) %Potencia normalizada por  usuario
ni=0;
end

figure
semilogy(SNR_dB,BER,'s-')
grid on
xlabel('SNR, (dB)');
ylabel('ABEP');
'''