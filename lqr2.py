import numpy as np
import matplotlib.pyplot as plt
# from numpy import kron, eye
# from numpy.linalg import norm
# from scipy.linalg import expm
import scipy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from logger import Logger
import torch



class LQRCompare():
    """docstring for QuantumLQR"""
    def __init__(self, s, q, T, L, N_Sample_J, seed=0, per_step=10, n_epoch=10000, name='classical'):
        args = locals()
        self.logger = Logger(name=name, path=None)
        self.logger.write_text("arguments ========")
        for k, v in args.items():
            if k == 'self':
                continue
            self.logger.write_text("{}: {}".format(k, v))

        np.random.seed(seed)
        self.p = {
            's' : s,
            'q' : q,
            'T' : T,
            'L' : L,
            'N_Sample_J': N_Sample_J,
            'per_step': per_step,
            'basis': 'Legendre',
            'n_epoch': n_epoch,
            'name': name
        }

    def run(self):
        self.A, self.B, self.Q, self.R = self.build_LQR_system(
            self.p['s'], self.p['q'], self.p['T'], self.p['L'])

        self.p['tau'] = self.p['T']
        self.p['n'] = 2 * self.p['s']
        self.p['m'] = self.p['s']
        self.p['N_Sample_loss'] = 20
        self.p['lr'] = 4e-1
        self.p['x_var'] = 1.0
        print("self.p", self.p)

        self.x0 = np.random.normal(0, self.p['x_var'], [self.p['n'], self.p['N_Sample_loss']]) 
        # self.build_quantum_system()
        self.compute_gt()

        K = np.random.normal(0, 1e-3, [self.p['m'], self.p['n']]) 
        self.K = torch.tensor(K, requires_grad=True)


        if self.p['name'] == 'quantum':
            self.optimize_quantum()
        elif self.p['name'] == 'classical':
            self.optimize_classical()
        else:
            self.optimize_DFO()

    def compute_gt(self):
        def lqr(A, B, Q, R):
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            return K
        K = lqr(self.A, self.B, self.Q, self.R)
        self.K = torch.tensor(K)
        J = self.eval_J()
        fK = self.eval_fK()
        self.gt_J = J
        self.gt_K = K
        self.gt_fK = fK
        print('gt J', J)
        print('gt fK', fK)
        print('gt K', K)

    def print_loss(self, i_epoch):

        J = self.eval_J()
        fK = self.eval_fK()

        st = "epoch: {:04d}, J: {}, loss_J: {}, loss_K: {}, loss_fK: {}".format(
            i_epoch, 
            J,
            J - self.gt_J,
            np.abs(self.K.detach().numpy() - self.gt_K).max(),
            fK - self.gt_fK
        )

        self.logger.write_text(st)
        return fK


    def optimize_classical(self):
        n_epoch = self.p['n_epoch']
        lr = self.p['lr']

        optimizer = torch.optim.Adam([self.K], lr=lr)

        for i_epoch in range(n_epoch):

            self.print_loss(i_epoch)
            optimizer.zero_grad()
            grad_K = self.classical_gradient_estimation()
            self.K.grad = grad_K
            optimizer.step()

    def optimize_DFO(self):
        n_epoch = self.p['n_epoch']
        lr = self.p['lr']
        self.i_epoch = 0
        with torch.no_grad():
            K0 = self.K.detach().numpy()
            def objective(K):
                self.K = torch.tensor(K.reshape(self.K.shape))
                fK = self.print_loss(self.i_epoch)
                self.i_epoch += 1
                return fK
            solution = scipy.optimize.minimize(objective, K0.reshape(-1), method=self.p['name'])
        

    def classical_gradient_estimation(self):
        tau = self.p['tau']
        n = self.p['n']
        m = self.p['m']
        N = n
        N_Sample_J = self.p['N_Sample_J']
        r = 1e-3

        K = np.matrix(self.K.detach().numpy())


        grad_K = np.zeros([m, n])

        for i in range(N):
            Ui = np.random.uniform(low=-1.0, high=1.0, size=[m*n])
            Ui = Ui / np.linalg.norm(Ui) * np.sqrt(m*n)
            Ui = np.reshape(Ui, [m, n])

            K1 = K + r * Ui
            K2 = K - r * Ui

            x0 = np.random.normal(0, self.p['x_var'], [n, 1]) 

            A_BK1 = self.A - self.B @ K1
            B_1 = self.Q + K1.T @ self.R @ K1
            A_BK2 = self.A - self.B @ K2
            B_2 = self.Q + K2.T @ self.R @ K2

            J1 = 0
            J2 = 0
            for i_J in range(N_Sample_J):
                t = tau / N_Sample_J * (i_J + 1)
                
                x = scipy.linalg.expm(A_BK1 * t) @ x0
                J1 += x.T @ B_1 @ x
                
                x = scipy.linalg.expm(A_BK2 * t) @ x0
                J2 += x.T @ B_2 @ x

            J1 = J1[0, 0] / N_Sample_J
            J2 = J2[0, 0] / N_Sample_J

            grad_K = grad_K + 1.0 / 2 / r / N * (J1 - J2) * Ui

        return torch.tensor(grad_K)



    def optimize_quantum(self):
        n_epoch = self.p['n_epoch']
        lr = self.p['lr']


        optimizer = torch.optim.Adam([self.K], lr=lr)

        for i_epoch in range(n_epoch):

            self.print_loss(i_epoch)

            optimizer.zero_grad()
            grad_K = self.quantum_gradient_estimation()
            self.K.grad = grad_K

            # if i_epoch % 10 == 0:
            #     print(grad_K)
            #     print(f"{i_epoch} -------")
            #     print(self.K.detach().numpy())
            optimizer.step()


    def eval_fK(self):

        tau = self.p['tau']
        n = self.p['n']
        m = self.p['m']

        K = np.matrix(self.K.detach().numpy())

        A_BK = self.A - self.B @ K

        P_ = self.Q + K.T @ self.R @ K


        def integrand(t, M):
            expm_t = scipy.linalg.expm(A_BK.T * t)
            expm = scipy.linalg.expm(A_BK * t)
            return expm_t @ M @ expm

        num_steps = self.p['N_Sample_J']

        # Time points for numerical integration
        t_points = np.linspace(0, tau, num_steps)
        dt = tau / num_steps

        # Initialize the result matrix
        P = np.matrix(np.zeros_like(A_BK))

        # Use the trapezoidal rule for integration
        for i in range(num_steps - 1):
            t1 = t_points[i]
            t2 = t_points[i + 1]
            P += (integrand(t1, P_) + integrand(t2, P_)) * dt / 2



        return np.trace(P)



    def eval_J(self):
        tau = self.p['tau']
        n = self.p['n']
        m = self.p['m']

        N_Sample_J = self.p['N_Sample_J']
        N_Sample_loss = self.p['N_Sample_loss']

        K = np.matrix(self.K.detach().numpy())
        A_BK = self.A - self.B @ K
        B_ = self.Q + K.T @ self.R @ K
        J = 0

        # x0 = np.random.normal(0, self.p['x_var'], [n, 1]) 
        # x0 = np.ones([n, 1]) * 3e-2
        x0 = self.x0
        for i_J in range(N_Sample_J):
            t = tau / N_Sample_J * (i_J + 1)
            x = scipy.linalg.expm(A_BK * t) @ x0
            J += np.trace(x.T @ B_ @ x)
        return J / N_Sample_J 



    def quantum_gradient_estimation(self):

        tau = self.p['tau']
        n = self.p['n']
        m = self.p['m']

        K = np.matrix(self.K.detach().numpy())

        A_BK = self.A - self.B @ K

        P_ = self.Q + K.T @ self.R @ K
        X_ = np.eye(A_BK.shape[0])


        def integrand(t, M):
            expm_t = scipy.linalg.expm(A_BK.T * t)
            expm = scipy.linalg.expm(A_BK * t)
            return expm_t @ M @ expm

        num_steps = self.p['N_Sample_J']

        # Time points for numerical integration
        t_points = np.linspace(0, tau, num_steps)
        dt = tau / num_steps

        # Initialize the result matrix
        P = np.matrix(np.zeros_like(A_BK))
        X = np.matrix(np.zeros_like(A_BK))

        # Use the trapezoidal rule for integration
        for i in range(num_steps - 1):
            t1 = t_points[i]
            t2 = t_points[i + 1]
            P += (integrand(t1, P_) + integrand(t2, P_)) * dt / 2
            X += (integrand(t1, X_) + integrand(t2, X_)) * dt / 2

        grad_K = 2 * (self.R @ K - self.B.T @ P) @ X

        return torch.tensor(grad_K)


    @staticmethod
    def build_LQR_system(s, q, T, L):        
        T = 2 * np.eye(s)
        T[:s-1, 1:s] -= np.eye(s-1)
        T[1:s, :s-1] -= np.eye(s-1) 

        #     x  xd
        # x   0   I
        # xd -T   -T
        A = np.zeros([2*s, 2*s])
        A[:s, s:] = np.eye(s)
        A[s:, :s] = -T
        A[s:, s:] = -T

        B = np.zeros([2*s, s])
        B[s:, :] = np.eye(s) 

        x = np.ones([2*s]) / 2*s

        Q = np.eye(2*s)
        Q[0, 0] = Q[0, 0] + 100.

        R = np.eye(s)
        R[1, 1] = R[1, 1] + 4.

        return np.matrix(A), np.matrix(B), np.matrix(Q), np.matrix(R)

if __name__ == '__main__':
    # sys = QuantumLQR(s=4, q=9, T=10, L=10)
    # name = 'SLSQP'
    name = 'quantum'
    # name = 'classical'
    sys = LQRCompare(s=4, q=9, T=10, L=10, seed=3, n_epoch=20000, N_Sample_J=8000, name=name)
    sys.run()
