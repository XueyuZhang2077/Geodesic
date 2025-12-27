import numpy as np
import sympy as sp




class Metric:
    def __init__(self,symbols_list,
                      G_expr):
        self._dim=len(symbols_list)

        self._symbols=symbols_list

        self._g=[ [self._expr_2_func(G_expr[i,j])  for j in range(i+1)] for i in range(self._dim)]

        partialk_gij=[[ [ sp.diff(G_expr[i,j],symbols_list[k]) for j in range(i+1) ] for i in range(self._dim)]  for k in range(self._dim) ]

        partiall_partialk_gij=[[[ [ sp.diff(partialk_gij[l][i][j],symbols_list[k]) for j in range(i+1) ] for i in range(self._dim)]  for k in range(l+1) ] for l in range(self._dim)]

        self._dg=[[ [ self._expr_2_func(partialk_gij[k][i][j]) for j in range(i+1) ] for i in range(self._dim)]  for k in range(self._dim) ]

        self._ddg=[[[ [ self._expr_2_func(partiall_partialk_gij[l][k][i][j]) for j in range(i+1) ] for i in range(self._dim)]  for k in range(l+1) ] for l in range(self._dim)]


    def _expr_2_func(self,expr):
        return sp.lambdify(self._symbols,expr,'numpy')

    def G(self,coords):
        cell_gi_gj=np.zeros((coords.shape[0],self._dim,self._dim),dtype=coords.dtype)
        for i in range(self._dim):
            for j in range(i+1):
                cell_gi_gj[:,i,j]=self._g[i][j](*coords.T)
                cell_gi_gj[:,j,i]=cell_gi_gj[:,i,j]
        return cell_gi_gj
    def dG(self,coords):
        cell_partial_gi_gj=np.zeros((coords.shape[0],self._dim,self._dim,self._dim),dtype=coords.dtype)
        for p in range(self._dim):
            for i in range(self._dim):
                for j in range(i+1):
                    cell_partial_gi_gj[:,p,i,j]=self._dg[p][i][j](*coords.T)
                    cell_partial_gi_gj[:,p,j,i]= cell_partial_gi_gj[:,p,i,j]
        return cell_partial_gi_gj
    
    def ddG(self,coords):
        cell_P_p_gi_gj=np.zeros((coords.shape[0],self._dim,self._dim,self._dim,self._dim),dtype=coords.dtype)
        for P in range(self._dim):
            for p in range(P+1):
                for i in range(self._dim):
                    for j in range(i+1):
                        cell_P_p_gi_gj[:,P,p,i,j]=self._ddg[P][p][i][j](*coords.T)
                        cell_P_p_gi_gj[:,P,p,j,i]=cell_P_p_gi_gj[:,P,p,i,j]
                cell_P_p_gi_gj[:,p,P]=cell_P_p_gi_gj[:,P,p]
        return cell_P_p_gi_gj
    


class Manifold:
    def __init__(self,
                 metric:Metric,
                 ):
        self._G=metric.G
        self._dG=metric.dG
        self._ddG=metric.ddG
        self._dim=metric._dim
        self._divide_num=None

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
        
        return Gammak_ij,np.append(partial_m__Gammak_ij,partial_m__Gammak_ij,axis=1)

    def _testfuncs_prep(self):

        self._delta_x=np.append(0.5*np.identity(self._dim,dtype=float),0.5*np.identity(self._dim,dtype=float),axis=0)
        self._delta_dx=np.append(-np.identity(self._dim,dtype=float),np.identity(self._dim,dtype=float),axis=0)

        deltaID_in_cell0=np.arange(0,self._dim*2,1,dtype=int)
        deltaID_partialID_in_cell0=np.array([[ [i,j] for j in range(self._dim*2)] for i in range(self._dim*2)])
        cell_deltaID_partialID=np.array([deltaID_partialID_in_cell0+i*self._dim for i in range(self._divide_num)])

        self._cell_deltadx_partialdx=np.einsum("c,di,pi->cdp",np.ones(self._divide_num,dtype=float),self._delta_dx,self._delta_dx)
        self._cell_delta_partialID_row=cell_deltaID_partialID[:,:,:,0].ravel()
        self._cell_delta_partialID_col=cell_deltaID_partialID[:,:,:,1].ravel()
        self._cell_deltaID=np.array([deltaID_in_cell0+i*self._dim for i in range(self._divide_num)]).ravel()
        

    def solve_geodesic(self,ini_line_node_coords,tol=1e-8):
        divide_num=ini_line_node_coords.shape[0]-1
        if divide_num!=self._divide_num:
            self._divide_num=divide_num
            self._testfuncs_prep()

        line=ini_line_node_coords.copy()
        free_ids=np.arange(self._dim,self._divide_num*self._dim,1,dtype=int)
        free2D_ids=np.ix_(free_ids,free_ids)

        cell_x=(line[:-1]+line[1:])/2.0
        cell_dx=line[1:]-line[:-1]
        cell_Gammak_i_j, cell_partial_Gammak_i_j=self._connection_coef_info(cell_x)
        cell_delta=np.einsum("ckij,ci,cj,dk->cd",cell_Gammak_i_j,cell_dx,cell_dx,self._delta_x)-np.einsum("di,ci->cd",self._delta_dx,cell_dx)
        delta=np.zeros(self._dim*(self._divide_num+1),dtype=float)
        np.add.at(delta,self._cell_deltaID,cell_delta.ravel())
        err=np.linalg.norm(delta[free_ids])
        print("err=",err)
        while np.linalg.norm(err)>tol:
            cell_delta_partial=2*np.einsum("ckij,ci,pj,dk->cdp",cell_Gammak_i_j,cell_dx,self._delta_dx,self._delta_x)+np.einsum("cpkij,ci,cj,dk->cdp",cell_partial_Gammak_i_j,cell_dx,cell_dx,self._delta_x)-self._cell_deltadx_partialdx
            delta_partial=np.zeros((delta.shape[0],delta.shape[0]),dtype=float)
            np.add.at(delta_partial,(self._cell_delta_partialID_row,self._cell_delta_partialID_col),cell_delta_partial.ravel())
            line[1:-1]-=(np.linalg.solve(delta_partial[free2D_ids],delta[free_ids])).reshape(line[1:-1].shape)

            cell_x=(line[:-1]+line[1:])/2.0
            cell_dx=line[1:]-line[:-1]
            cell_Gammak_i_j, cell_partial_Gammak_i_j=self._connection_coef_info(cell_x)
            cell_delta=np.einsum("ckij,ci,cj,dk->cd",cell_Gammak_i_j,cell_dx,cell_dx,self._delta_x)-np.einsum("di,ci->cd",self._delta_dx,cell_dx)
            delta=np.zeros(self._dim*(self._divide_num+1),dtype=float)
            np.add.at(delta,self._cell_deltaID,cell_delta.ravel())
            err=np.linalg.norm(delta[free_ids])
            print("err=",err)
        return line







if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x,y=sp.symbols("x y",positive=True)
    r=sp.sqrt(x**2+y**2)
    th=sp.atan(y/x)
    p=sp.Matrix([r,th])
    p_x=sp.diff(p,x)
    p_y=sp.diff(p,y)
    g_xx=sp.simplify(p_x.dot(p_x))
    g_xy=sp.simplify(p_x.dot(p_y))
    g_yy=sp.simplify(p_y.dot(p_y))
    g=sp.Matrix([[g_xx,g_xy],
                 [g_xy,g_yy]])
    
    metric=Metric([x,y],g)
    manifold=Manifold(metric)
    ini_line=np.c_[np.linspace(1.0,0.0,101),np.linspace(0.0,1.0,101)]
    line=manifold.solve_geodesic(ini_line)
    fig=plt.figure()
    ax=fig.add_subplot(121)
    ax.set(xlim=[-1.1,1.1],ylim=[-1.1,1.1],aspect=1)
    ax.plot(np.sqrt(line[:,0]**2+line[:,1]**2),np.arctan(line[:,1]/line[:,0])) #in real manifold it is a straight line
    ax.set_title("real manifold")

    bx=fig.add_subplot(122)
    bx.set(xlim=[-1.1,1.1],ylim=[-1.1,1.1],aspect=1)
    bx.plot(line[:,0],line[:,1]) #but in the chart (or map) (x,y), it appears like a curved line
    bx.set_title("map for the manifold")
    plt.show()
