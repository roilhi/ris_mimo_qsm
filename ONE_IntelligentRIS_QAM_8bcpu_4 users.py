import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from comm_utilities import qam_symbol_generator, biterr_calculation, qam_mimo_tx_combinations, H_channel, awgn_noise

Nr = 2
Nt = 8 
Nrel = 32
SNR_dB = np.arange(start=-10, stop=32, step=2)
SNR_l = 10**(SNR_dB/10)
be = 0
BER = np.empty([len(SNR_dB),])

M = 16
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

SS = qam_mimo_tx_combinations(Nr, qam_sym)
tam_SS = int(len(SS))
norma = np.abs(SS)
norma2 = norma[:,0] + norma[:,1]
pot = np.sum(norma2)/256
SS /= pot
n = 0
P_prom_real = 0
N_iter =  10**3
P = 1
At = np.sqrt(P)
bcpu = int(np.log2(tam_SS))

for j, snr in enumerate(SNR_l):
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
        # Channel H
        H_SR1 = H_channel(Nrel,8)
        # Relay channels for each user
        H11 = H_channel(2,Nrel)
        H21 = H_channel(2,Nrel)
        H31 = H_channel(2,Nrel)
        H41 = H_channel(2,Nrel)
        #Direct path
        H_SD1 = H_channel(2,Nt)
        H_SD2 = H_channel(2,Nt)
        H_SD3 = H_channel(2,Nt)
        H_SD4 = H_channel(2,Nt)
        # Composed channel from RIS 1 to all users
        H_RD1 = np.vstack((H11,H21,H31,H41))
        # Direct path matrix to all users
        H_SD = np.vstack((H_SD1,H_SD2,H_SD3,H_SD4))
        ##########################################
        # OPTIMIZATION BY GENETIC ALGORITHM
        # #######################################
        N = 12 # number of random vectors
        Z = 40 # Number of iterations
        cualidad = np.ones((N,1))
        phi_todos = np.ones((N,Nrel), dtype='complex')
        hijo = np.empty_like(phi_todos)
        Max = 0 # original power
        for h in range(Z):
            for n in range(N):
                # phases in the RIS
                if h==0:
                    phi = (1/np.sqrt(2))*(np.random.standard_normal(size=(Nrel,))+1j*(np.random.standard_normal(size=(Nrel,))))
                    phi /= np.abs(phi)
                else:
                    phi = hijo[n,:]
                Phi = np.diag(phi)
                # System equivalent matrix
                Heq = H_RD1 @ Phi @ H_SR1 + H_SD
                # equivalent channels for each user
                H1 = (H11 @ Phi @ H_SR1) + H_SD1
                H2 = (H21 @ Phi @ H_SR1) + H_SD2
                H3 = (H31 @ Phi @ H_SR1) + H_SD3
                H4 = (H41 @ Phi @ H_SR1) + H_SD4

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
                Tx *= P*np.sqrt(snr)
                # RIS reception
                Rx1 = (1/At)*(H_SR1 @ Tx)
                Tx2 = np.expand_dims(phi,axis=0).T * Rx1
                noise = awgn_noise(1,Nr)
                # Reception from user 1
                S_y2 = (1/At)*(H11 @ Tx2) + H_SD1 @ Tx
                y2 = S_y2 + noise
                cualidad[n,:] = np.linalg.norm(S_y2)
                phi_todos[n,:] = phi
            # Selection
            index = np.argmax(cualidad)
            papa = phi_todos[index,:]
            cualidad[index] = 0
            index = np.argmax(cualidad)
            mama = phi_todos[index,:]
            cualidad[index] = 0
            index = np.argmax(cualidad)
            papa2 = phi_todos[index,:]
            cualidad[index] = 0
            index = np.argmax(cualidad)
            mama2 = phi_todos[index,:]
            # Cruza
            hijo[0,:] = np.hstack((papa[:16],mama[16:]))
            hijo[1,:] = np.hstack((papa[:16], mama2[16:]))
            hijo[2,:] = np.hstack((papa2[:16], mama[16:]))
            hijo[3,:] = np.hstack((papa2[:16],mama2[16:]))
            hijo[4,:] = np.hstack((mama[:16],papa[16:]))
            hijo[5,:] = np.hstack((mama[:16], papa2[16:]))
            hijo[6,:] = np.hstack((mama2[:16],papa[16:]))
            hijo[7,:] = np.hstack((mama2[:16],papa2[16:])) 
            # Mutacion
            for k in range(Nt):
                muta = (1/np.sqrt(2))*(np.random.standard_normal()+1j*np.random.standard_normal())
                muta /= np.abs(muta)
                hijo[k,np.random.randint(32)] = muta 
            N = 8
        # end optimization
        # using the best selected phase vector for the RIS
        phi = papa
        Phi = np.diag(phi)
        # Composed channel for RIS 1 to all users
        H_RD1 = np.vstack((H11,H21,H31,H41))
        # Composed matrix for direct path to all users
        H_SD = np.vstack((H_SD1,H_SD2,H_SD3,H_SD4))
        # equivalent system matrix
        Heq = H_RD1 @ Phi @ H_SR1 + H_SD 
        # equivalent channels for each user
        H1 = (H11 @ Phi @ H_SR1) + H_SD1
        H2 = (H21 @ Phi @ H_SR1) + H_SD2
        H3 = (H31 @ Phi @ H_SR1) + H_SD3
        H4 = (H41 @ Phi @ H_SR1) + H_SD4 
        # auxiliar matrices for BD
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
        Tx *= P*np.sqrt(snr)
        # RIS reception
        Rx1 = (1/At)*(H_SR1 @ Tx)
        # Transmission from RIS
        Tx2 = np.expand_dims(phi,axis=1)*Rx1
        noise = awgn_noise(1,Nr)
        # reception from user 1
        S_y2 = (1/At) * (H11 @ Tx2) + H_SD1 @ Tx
        y2 = S_y2 + noise 

        C2 = H1 @ W1
        C = SS @ C2.T
        # ML detector
        s1 = np.square(np.abs(y2[0]-np.sqrt(snr)*C[:,0]))
        s2 = np.square(np.abs(y2[1]-np.sqrt(snr)*C[:,1]))
        # MRC combiner
        s = s1 + s2
        index = np.argmin(s)

        if index != dato:
            a = biterr_calculation(dato, index, bcpu)
            be += a
    BER[j] = be/(N_iter*bcpu)
    be = 0
    print(f'BER = {BER[j]}')

fig,ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(SNR_dB,BER,color='r',marker='o',linestyle='-')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('ABEP')
ax.set_title('ABEP 1 RIS 4 users 8 bcpu QAM GA')
plt.grid()
plt.show()






