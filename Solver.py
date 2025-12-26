import numpy as np
import scipy.sparse as sp


class Manifold:
    def __init__(self,
                 metric,
                 dmetric,
                 ddmetric,
                 dim,
                 free_node_num_for_geodesic=100):
        self._G=metric
        self._dG=dmetric
        self._ddG=ddmetric
        self._dim=dim
        self._test_func_num=free_node_num_for_geodesic*dim
        self._testfuncs_prep()

    def _connection_coef_info(self,x):
        g=self._G(x)
        gI=np.linalg.inv(g)
        partial_g=self._dG(x)
        partial2_g=self._ddG(x)

        Gamma_lij=(np.einsum("...ilj->...lij",partial_g)+np.einsum("...jli->...lij",partial_g)-partial_g)/2
        Gammak_ij=np.einsum("...kl,...lij->...kij",gI,Gamma_lij)
        partial_gI=-np.einsum("...ik,...jl,...mkl->...mij",gI,gI,partial_g)
        temp_mlij=(np.einsum("...milj->...mlij",partial2_g)+np.einsum("...mjli->...mlij",partial2_g)-partial2_g)/2
        partial_m__Gammak_ij=np.einsum("...mkl,...lij->...mkij",partial_gI,Gamma_lij)+np.einsum("...kl,...mlij->...mkij",gI,temp_mlij)

        return Gammak_ij,partial_m__Gammak_ij

    def _testfuncs_prep(self):
        test_func_num,dim=self._test_func_num,self._dim
        #row,col ids for test funcs
        row=np.arange(0,test_func_num,1,dtype=int)
        col=row.copy()
        row=np.append(row,row)
        col=np.append(col,col+dim)

        self._delta_x=sp.coo_matrix((np.full(test_func_num*2,0.5),(row,col)),(test_func_num,test_func_num+dim))
        self._delta_dx=sp.coo_matrix(
            (np.append(np.ones(test_func_num),-np.ones(test_func_num)),(row,col)),
            (test_func_num,test_func_num+dim)
        )
        self._delta_dx_dot_dx=self._delta_dx.dot(self._delta_dx.T)
        print(self._delta_dx_dot_dx.toarray())
        # self._delta_dx_Ddx=identitymatrix

        #row,col ids for jacobi of Gamma^i_jk dx^j dx^k, etc.
        row=np.array([np.full(dim,i) for i in range(test_func_num)])
        col=np.einsum("...ij->...ji",row)
        row,col=row.ravel(),col.ravel()

        self._row=np.append(row,row+dim)
        self._col=np.append(col,col)


    def solve_geodesic(self,ini_line_node_coords,tol=1e-8):
        line=ini_line_node_coords.copy()
        free_shape=line[1:-1].shape

        x=(line[:-1]+line[1:])/2.0
        dx=line[1:]-line[:-1]
        Gamma, partial_Gamma=self._connection_coef_info(x)
        K,M=2*np.einsum("...ijk,...k->...ij",Gamma,dx),np.einsum("...lijk,...j,...k->...il",partial_Gamma,dx,dx)/2
        v=np.einsum("...ij,...j->...i",K/2,dx)
        err=self._delta_dx.dot(dx.ravel())-self._delta_x.dot(v.ravel())
        print("err=",np.linalg.norm(err))
        while np.linalg.norm(err)>tol:

            K=sp.coo_matrix((np.append(K[:-1].ravel(),-K[1:].ravel()),(self._row,self._col)),(self._test_func_num+self._dim,self._test_func_num))
            M=sp.coo_matrix((np.append(M[:-1].ravel(),M[1:].ravel()),(self._row,self._col)),(self._test_func_num+self._dim,self._test_func_num))
            Jacob=(self._delta_dx_dot_dx-self._delta_x.dot(K+M)).toarray()
            line[1:-1]=(line[1:-1].ravel()-np.linalg.inv(Jacob)@err).reshape(free_shape)

            x = (line[:-1] + line[1:]) / 2
            dx = line[1:] - line[:-1]
            Gamma, partial_Gamma = self._connection_coef_info(x)
            K, M = 2 * np.einsum("...ijk,...k->...ij", Gamma, dx), np.einsum("...lijk,...j,...k->...il", partial_Gamma,
                                                                             dx, dx) / 2
            v = np.einsum("...ij,...j->...i", K / 2, dx)
            err = self._delta_dx.dot(dx.ravel()) + self._delta_x.dot(v.ravel())
            print("err=", np.linalg.norm(err))

        return line










if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def G(coords):
        r=coords[:,0]
        g=np.zeros((coords.shape[0],2,2),dtype=float)
        g[:,0,0]=1.0
        g[:,1,1]=r**2
        return g
    def dG(coords):
        r=coords[:,0]
        dg=np.zeros((coords.shape[0],2,2,2),dtype=float)

        dg[:,0,1,1]=2*r

        return dg
    def ddG(coords):
        r=coords[:,0]
        ddg=np.zeros((coords.shape[0],2,2,2,2),dtype=float)
        ddg[:,0,0,1,1]=2
        return ddg

    manifold=Manifold(G,dG,ddG,2,50)
    ini_line=np.c_[np.linspace(1.0,1.0,52),np.linspace(-np.pi/2,-np.pi/4,52)]
    line=manifold.solve_geodesic(ini_line)
    plt.plot(line[:,0]*np.cos(line[:,1]),line[:,0]*np.sin(line[:,1]))
    plt.show()