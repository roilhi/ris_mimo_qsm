import numpy as np
import matplotlib.pyplot as plt
from comm_utilities import *
from scipy.linalg import svd

pot=0
Nr=2 #Antenas por usuario
Nt=8 # Tx en la Estacion Base
Nrel=16 # Número de espejos en el RIS( mayor que 8)
SNR_dB = np.arange(start=0,stop=32,step=2) #start=0, stop = 20, step = 2 vector SNR de ruido en dB
SNR_l = 10**(SNR_dB/10) # snr lineal
be = 0
#BER = np.zeros((1,len(SNR_dB)))
BER = []
M = 16 #i=0:M-1;
#i = np.arange(M)
#f = QAMModem(M) # crea un objeto modulador QAM de 16 símbolos
#QAM_sym = f.modulate(dec2bitarray(i,4)) # crea los símbolos QAM, dec2bitarray convierte de entero a bit (int)

#idx_bin = [2,3,1,0,6,7,5,4,14,15,13,12,10,11,9,8]

#QAM_sym = np.array([QAM_sym[idM] for idM in idx_bin]) # cambiar los símbolos al orden natural MATLAB

QAM_sym = np.array(qam_symbol_generator(M))
FN = 1/np.sqrt((2/3)*(M-1)) #FN=1/sqrt((2/3)*(M-1)); # ¿Constante de Canal?
QAM_sym = QAM_sym*FN
suma = 0
for a in range(16):
    pot1 = np.sqrt(np.real(QAM_sym[a])**2+np.imag(QAM_sym[a])**2)
    suma += pot1
pot = suma/M # Potencia promedio de la constelación
QAM_symN = QAM_sym/pot #Normalización por antena
QAM_symNU =  QAM_symN/2 #Normalización por usuario
#dato = np.random.randint(M)

#SS = np.array([0,0]) # para concatenar con array de símbolos
#for A1 in range(M):
#    for A2 in range(M):
#        SS = np.vstack((SS,np.array([QAM_symNU[A1],QAM_symNU[A2]])))

#SS = np.delete(SS,(0), axis=0)

SS = qam_mimo_tx_combinations(Nr,QAM_symNU)

P_t = 0
len_data = int(len(SS))
bcpu = int(np.log2(len_data))
cst_ch = 1/np.sqrt(2)
n_iter = 10**5
P = 10
At = np.sqrt(P)
#for j in range(len(SNR_l)):
for snr in SNR_l:
    P_acu = 0
    for i in range(n_iter):
        # datos a transmitir
        cst_ch = 1/np.sqrt(2)
        #x_vec = np.array([]) #vector de transmisión
        # Canal ---H-----
        H_sr = cst_ch*(np.random.randn(Nrel,8)+np.random.randn(Nrel,8)*1j) #relays independientes
        #datos = np.random.randint(256, size=(4))
        dato = np.random.randint(len_data)
        dato2 = np.random.randint(len_data)
        dato3 = np.random.randint(len_data)
        dato4 = np.random.randint(len_data)
        x1 = SS[dato,:]
        x2 = SS[dato2,:]
        x3 = SS[dato3,:]
        x4 = SS[dato4,:]
        #x_vec = np.vstack([SS[elem].reshape(2,1) for elem in datos])
        x_vec = np.hstack((x1.T,x2.T,x3.T,x4.T)).reshape(8,1)
        # Genera el canal de relay por usuario
        H12 = cst_ch*(np.random.randn(2,Nrel)+np.random.randn(2,Nrel)*1j)
        H22 = cst_ch*(np.random.randn(2,Nrel)+np.random.randn(2,Nrel)*1j)
        H32 = cst_ch*(np.random.randn(2,Nrel)+np.random.randn(2,Nrel)*1j)
        H42 = cst_ch*(np.random.randn(2,Nrel)+np.random.randn(2,Nrel)*1j)
        H_RD = np.vstack((H12,H22,H32,H42))
        #H_RD = cst_ch*(np.random.randn(8,Nrel)+(np.random.randn(8,Nrel))*1j)
        Heq = np.matmul(H_RD, H_sr)
        # Otra forma para calcular canales equivalentes para cada usuario
        # H1 = Heq[0:2,:]
        # H2 = Heq[2:4,:]
        # H3 = Heq[4:6,:]
        # H4 = Heq[6:8,:]
        # -----  H = [np.matrix(H_RD[k,l])*np.matrix(H_sr) for k,l in zip(range(0,len(H_RD)-1,2),range(2,len(H_RD)+1,2)]
        #H = [Heq[k:l] for k,l in zip(range(0,len(Heq)-1,2),range(2,len(Heq)+1,2))]
        H1 = np.matmul(H12,H_sr)
        H2 = np.matmul(H22,H_sr)
        H3 = np.matmul(H32,H_sr)
        H4 = np.matmul(H42,H_sr)
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
        #idx_Hc = np.array([[2,3,4],[1,3,4],[1,2,4],[1,2,3]])-1 # Para obtener Hc1, Hc2, Hc3, Hc4
        #Hc = [np.array([H[elem] for elem in k]) for k in idx_Hc] # Lista con Hc1, Hc2, Hc3, Hc4
        #W = []
        #Tx = []
        #for k in range(len(Hc)):
        #    dim = Hc[k].shape
        #    Hcrep = np.reshape(Hc[k],(dim[0]*dim[1],dim[2])) # Hc toma dimensiones (3,2,8), debe reducirse a 2 dimensiones 
        #    U,S,VT = svd(Hcrep) # svd del elemento redimensionado a (6,8)
        #    W.append(VT[:,len(VT)-2:len(VT)])
        Tx = np.matmul(W1,x_vec[0:2]) + np.matmul(W2,x_vec[2:4]) + np.matmul(W3,x_vec[4:6]) + np.matmul(W4,x_vec[6:8])
        #Tx = sum([np.dot(W[e],x_vec[j:k]) for e in range (len(W)) for j,k in zip(range(0,len(x_vec)-1,2),range(2,len(x_vec),2))])
        P_acu += sum(abs(Tx))
        #Tx *= (P*np.sqrt(SNR_l[j]))
        #P = 10
        #At = np.sqrt(P)
        Tx = P*np.sqrt(snr)*Tx
        norma = np.abs(Tx)
        P_t2 = np.sum(norma)/4
        P_t += P_t2
        # Creando el ruido AWGN
        Rx1 = (1/At)*np.matmul(H_sr,Tx) # recepción en el RIS 
        Tx2 = Rx1 
        # Ruido en el receptor 1
        n2  = (1/np.sqrt(2))*np.random.randn(1,2)+(1/np.sqrt(2))*np.random.randn(1,2)*1j
        #n2 = 0.0
        # Recepción usuario 1
        #H12 = H_RD[:2]
        y2 = (1/At)*np.matmul(H12,Tx2)+n2.T
        C2 = np.matmul(H1,W1)
        C = np.matmul(SS,C2.T)
        #s1 = np.square(np.abs(y2[0]-np.sqrt(SNR_l[j])*C[:,0]))
        #s2 = np.square(np.abs(y2[1]-np.sqrt(SNR_l[j])*C[:,1]))
        s1 = np.square(np.abs(y2[0]-np.sqrt(snr)*C[:,0])) # señal recibida - vectores de comparación
        s2 = np.square(np.abs(y2[1]-np.sqrt(snr)*C[:,1])) 
        s = s1+s2
        #index = np.argwhere(s==np.argmin(s))[0]
        #index = np.where(s == s.min())[0][0] 
        index = np.argmin(np.abs(s))
        #index = index.min()
        # cuenta los errores de bit
        if index != dato:
            #dato_bit = dec2bitarray(dato,8)
            #index_bit = dec2bitarray(index,8)
            #a = len(np.argwhere(dato_bit != index_bit))
            #if a.size == 0:
            #    a = 0
            #else:
            #    a = a[0]
            a = biterr_calculation(dato,index,bcpu)
            be += a # Fin if
        # Fin for 2000
    #BER[j] = be/(i*8)
    BER.append(be/(n_iter*bcpu))
    be = 0 
    Ptot = P_acu/i 
# Fin for SNR
    print(Ptot)
    print(f' BER = {BER}\n')
fig,ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(SNR_dB,BER,color='r',marker='o',linestyle='-')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('ABEP')
plt.grid()
plt.show()