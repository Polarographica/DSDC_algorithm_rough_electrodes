# -*- coding: utf-8 -*-
"""

@author: Alex Tichter, Tim Tichter
"""

import numpy as np
from numba import jit, prange

class ThreeDimDerivFactory:
    def __init__(self, nopython=True, parallel=False, dtype=np.float32):
        self.NOPYTHON = nopython
        self.PARALLEL = parallel
        self.NOGIL = True
        self.DTYPE = dtype

        self.run_derimat = None
        self.run_dotprod = None

        self.run_three_dim_deriv_x = None
        self.run_three_dim_deriv_y = None
        self.run_three_dim_deriv_z = None

        self.run_function = None

        self.initialize_functions()

    def return_run_function(self):
        return self.run_function

    def initialize_functions(self):
        # Initialize the functions
        self.run_derimat = self.get_run_derimat()
        self.run_dotprod = self.get_run_dotprod()

        self.run_three_dim_deriv_x = self.get_run_three_dim_deriv_x()
        self.run_three_dim_deriv_y = self.get_run_three_dim_deriv_y()
        self.run_three_dim_deriv_z = self.get_run_three_dim_deriv_z()

        self.run_function = self.get_run_three_dim_deriv()

    def get_run_three_dim_deriv(self):
        dtype = self.DTYPE
        run_three_dim_deriv_x = self.run_three_dim_deriv_x
        run_three_dim_deriv_y = self.run_three_dim_deriv_y
        run_three_dim_deriv_z = self.run_three_dim_deriv_z
        #==========================================================================================================================
        #This function computes the three dimensional derivative. It calls the DotProd and the Matbuilder function at
        #iterates through the entire stack
        #==========================================================================================================================
        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL)
        def ThreeDimDeriv(stack, START):
            deriv_stack = np.zeros_like(stack, dtype=dtype)
            # for i in prange(len(stack[::,0,0])):
            #     for j in prange(len(stack[0,::,0])):
            #         aa,bb,cc = DeriMat(START[i,j,::])
            #         deriv_stack[i,j,::] += DotProd(aa,bb,cc,stack[i,j,::])
            # for i in prange(len(stack[::,0,0])):
            #     for k in prange(len(stack[0,0,::])):
            #         aa,bb,cc = DeriMat(START[i,::,k])
            #         deriv_stack[i,::,k] += DotProd(aa,bb,cc,stack[i,::,k])
            # for j in prange(len(stack[0,::,0])):
            #     for k in prange(len(stack[0,0,::])):
            #         aa,bb,cc = DeriMat(START[::,j,k])
            #         deriv_stack[::,j,k] += DotProd(aa,bb,cc,stack[::,j,k])
            
            deriv_stack = run_three_dim_deriv_x(stack, START, deriv_stack)
            deriv_stack = run_three_dim_deriv_y(stack, START, deriv_stack)
            deriv_stack = run_three_dim_deriv_z(stack, START, deriv_stack)

            return deriv_stack.sum()
        return ThreeDimDeriv
    
    def get_run_three_dim_deriv_x(self):
        DeriMat = self.run_derimat
        DotProd = self.run_dotprod

        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL, parallel=self.PARALLEL)
        def ThreeDimDerivX(stack, START, deriv_stack):
            for j in prange(len(stack[0,::,0])):
                for k in prange(len(stack[0,0,::])):
                    aa,bb,cc = DeriMat(START[::,j,k])
                    deriv_stack[::,j,k] += DotProd(aa,bb,cc,stack[::,j,k])
            return deriv_stack

        return ThreeDimDerivX

    def get_run_three_dim_deriv_y(self):
        DeriMat = self.run_derimat
        DotProd = self.run_dotprod

        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL, parallel=self.PARALLEL)
        def ThreeDimDerivY(stack, START, deriv_stack):
            for i in prange(len(stack[::,0,0])):
                for k in prange(len(stack[0,0,::])):
                    aa,bb,cc = DeriMat(START[i,::,k])
                    deriv_stack[i,::,k] += DotProd(aa,bb,cc,stack[i,::,k])
            return deriv_stack

        return ThreeDimDerivY
    
    def get_run_three_dim_deriv_z(self):
        DeriMat = self.run_derimat
        DotProd = self.run_dotprod

        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL, parallel=self.PARALLEL)
        def ThreeDimDerivZ(stack, START, deriv_stack):
            for i in prange(len(stack[::,0,0])):
                for j in prange(len(stack[0,::,0])):
                    aa,bb,cc = DeriMat(START[i,j,::])
                    deriv_stack[i,j,::] += DotProd(aa,bb,cc,stack[i,j,::])
            return deriv_stack
        
        return ThreeDimDerivZ
    
    def get_run_derimat(self):
        dtype = self.DTYPE

        #==========================================================================================================================
        #This funtion computes the matrix for a derivative along one dimension in a stack which is passed to it. it returns three
        #vectors which are passed to the DotProd function. Also, by being multiplied with the mirror-image of the initial stack
        # i.e. (START[..,..,..]-1)**2 it only counts the derivative at electrode-voxels. However, since two neighbouring electrolyte
        # voxels are at the same value, their derivative is zero. Hence, this function only counts electrode-electrolyte-boundaries.
        #==========================================================================================================================
        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL)
        def DeriMat(Array_Vector):
            lenx           = len(Array_Vector)
            Middle         = -2*np.ones(lenx, dtype=dtype)
            Middle[0]      = -1
            Middle[-1]     = -1
            OffDiag        = np.ones(lenx-1, dtype=dtype)
            Middle         = Middle*(Array_Vector-1)**2
            low1, up1      = OffDiag*(Array_Vector[1::]-1)**2, OffDiag*(Array_Vector[:-1:]-1)**2
            return low1,  Middle, up1
        
        return DeriMat
    
    def get_run_dotprod(self):
        #==========================================================================================================================
        #The DotProd function computed the dot product of a tridiagonal banded matrix with a vector. The matrix consists of three 
        #individual vectors a,b and c, which are the lower, central and upper diagonal, respectively. The classical np.dot does only 
        #work for dense matrices which would be a suuuuper waste of memory. This sparse tridiagonal dot product
        #stores the main diagonals only and is therefore much more efficient.
        #==========================================================================================================================
        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL)
        def DotProd(a,b,c,vect):
            Res = b*vect
            Res[1::]  += a*vect[:-1:] 
            Res[:-1:] += c*vect[1::]
            return Res
        
        return DotProd
