# -*- coding: utf-8 -*-
"""

@author: Alex Tichter
"""
import numpy as np
from numba import jit, prange


class DouglasGunnLoopFactory:
    def __init__(self, nopython=True, parallel=True, dtype=np.float32):
        self.NOPYTHON = nopython
        self.PARALLEL = parallel
        self.NOGIL = True
        self.DTYPE = dtype

        # Variables for the loop functions
        self.run_function = None
        self.run_one_third = None
        self.run_two_third = None
        self.run_three_third = None

        self.run_matbuilder = None
        self.run_matbuilder_z = None
        self.run_thomas = None

        self.initialize_functions()

    def return_run_function(self):
        return self.run_function

    def initialize_functions(self):
        # Initialize the functions
        self.run_matbuilder = self.get_run_matbuilder()
        self.run_matbuilder_z = self.get_run_matbuilder_z()
        self.run_thomas = self.get_run_thomas()

        self.run_direction_x = self.get_run_direction_x()
        self.run_direction_y = self.get_run_direction_y()
        self.run_direction_z = self.get_run_direction_z()

        self.run_function = self.get_run_function()
        
    def get_run_function(self):
        run_direction_x = self.run_direction_x
        run_direction_y = self.run_direction_y
        run_direction_z = self.run_direction_z

        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL)
        def DouglasGunnLoop(stack, START, ll, DiffKoeff, TimeIncr, Universal_LD_z, Universal_MD_z, Universal_UD_z, alphas_r, betas_r, gammas_r):
            Interm_Array_1  = np.ones_like(stack)
            stack_one_Third = np.ones_like(stack)
            stack_two_Third = np.ones_like(stack)
            #stack_complete  = np.ones_like(stack) #not needed, repurpose one of the earlier arrays
            #===========================================================================================================================
            #Timestep 1-->1/3
            #===========================================================================================================================
            Interm_Array_1[1:-1:,1:-1:,1:-1:] = (ll*(stack[0:-2:,1:-1:,1:-1:] + stack[2::,1:-1:,1:-1:] 
                                                + 2*(stack[1:-1:,0:-2:,1:-1:] + stack[1:-1:,2::,1:-1:])
                                                - 6*stack[1:-1:,1:-1:,1:-1:]) 
                                                + 2*DiffKoeff*TimeIncr*alphas_r*stack[1:-1:,1:-1:,0:-2:]  
                                                + 2*DiffKoeff*TimeIncr*betas_r*stack[1:-1:,1:-1:,1:-1:]  
                                                + 2*DiffKoeff*TimeIncr*gammas_r*stack[1:-1:,1:-1:,2::] 
                                                + 2*stack[1:-1:,1:-1:,1:-1:])*START 
            

            stack_one_Third = run_direction_x(START, stack_one_Third, ll, Interm_Array_1)

            stack_one_Third[ 0,1:-1:,1:-1:]           = stack_one_Third[ 1,1:-1:,1:-1:]
            stack_one_Third[-1,1:-1:,1:-1:]           = stack_one_Third[-2,1:-1:,1:-1:]
            stack_one_Third[1:-1:, 0,1:-1:]           = stack_one_Third[1:-1:, 1,1:-1:]
            stack_one_Third[1:-1:,-1,1:-1:]           = stack_one_Third[1:-1:,-2,1:-1:]
            stack_one_Third[1:-1:,1:-1:, 0]           = stack_one_Third[1:-1:,1:-1:, 1]
            stack_one_Third[1:-1:,1:-1:,-1]           = stack_one_Third[1:-1:,1:-1:,-2]
            #===========================================================================================================================
            #Timestep 1/3-->2/3
            #===========================================================================================================================
            Interm_Array_1[1:-1:,1:-1:,1:-1:] = (ll*(   stack[0:-2:,1:-1:,1:-1:] + stack[2::,1:-1:,1:-1:]
                                                    +  stack[1:-1:,0:-2:,1:-1:] + stack[1:-1:,2::,1:-1:]
                                                    -4*stack[1:-1:,1:-1:,1:-1:]
                                                    +  stack_one_Third[0:-2:,1:-1:,1:-1:]
                                                    -2*stack_one_Third[1:-1:,1:-1:,1:-1:]
                                                    +  stack_one_Third[2::,1:-1:,1:-1:])
                                                    +2*DiffKoeff*TimeIncr*alphas_r*stack[1:-1:,1:-1:,0:-2:]  
                                                    +2*DiffKoeff*TimeIncr*betas_r*stack[1:-1:,1:-1:,1:-1:]  
                                                    +2*DiffKoeff*TimeIncr*gammas_r*stack[1:-1:,1:-1:,2::] 
                                                    +2*stack[1:-1:,1:-1:,1:-1:])*START

            stack_two_Third = run_direction_y(START, stack_two_Third, ll, Interm_Array_1)
            stack_two_Third[ 0,1:-1:,1:-1:]           = stack_two_Third[ 1,1:-1:,1:-1:]
            stack_two_Third[-1,1:-1:,1:-1:]           = stack_two_Third[-2,1:-1:,1:-1:]
            stack_two_Third[1:-1:, 0,1:-1:]           = stack_two_Third[1:-1:, 1,1:-1:]
            stack_two_Third[1:-1:,-1,1:-1:]           = stack_two_Third[1:-1:,-2,1:-1:]
            stack_two_Third[1:-1:,1:-1:, 0]           = stack_two_Third[1:-1:,1:-1:, 1]
            stack_two_Third[1:-1:,1:-1:,-1]           = stack_two_Third[1:-1:,1:-1:,-2]
            #===========================================================================================================================
            #Timestep 2/3-->3/3
            #===========================================================================================================================
            Interm_Array_1[1:-1:,1:-1:,1:-1:] = (ll*(   stack[0:-2:,1:-1:,1:-1:] + stack[2::,1:-1:,1:-1:]
                                                    +  stack[1:-1:,0:-2:,1:-1:] + stack[1:-1:,2::,1:-1:]
                                                    -4*stack[1:-1:,1:-1:,1:-1:] 
                                                    +  stack_one_Third[0:-2:,1:-1:,1:-1:]
                                                    -2*stack_one_Third[1:-1:,1:-1:,1:-1:]
                                                    +  stack_one_Third[2::,1:-1:,1:-1:]
                                                    +  stack_two_Third[1:-1:,0:-2:,1:-1:]
                                                    -2*stack_two_Third[1:-1:,1:-1:,1:-1:]
                                                    +  stack_two_Third[1:-1:,2::,1:-1:])
                                                    +  DiffKoeff*TimeIncr*alphas_r*stack[1:-1:,1:-1:,0:-2:]  
                                                    +  DiffKoeff*TimeIncr*betas_r*stack[1:-1:,1:-1:,1:-1:]  
                                                    +  DiffKoeff*TimeIncr*gammas_r*stack[1:-1:,1:-1:,2::] 
                                                    +2*stack[1:-1:,1:-1:,1:-1:])*START

            Interm_Array_1 = run_direction_z(START, Interm_Array_1, Universal_LD_z, Universal_MD_z, Universal_UD_z, DiffKoeff, TimeIncr)
            Interm_Array_1[ 0,1:-1:,1:-1:]           = Interm_Array_1[ 1,1:-1:,1:-1:]
            Interm_Array_1[-1,1:-1:,1:-1:]           = Interm_Array_1[-2,1:-1:,1:-1:]
            Interm_Array_1[1:-1:, 0,1:-1:]           = Interm_Array_1[1:-1:, 1,1:-1:]
            Interm_Array_1[1:-1:,-1,1:-1:]           = Interm_Array_1[1:-1:,-2,1:-1:]
            Interm_Array_1[1:-1:,1:-1:, 0]           = Interm_Array_1[1:-1:,1:-1:, 1]
            Interm_Array_1[1:-1:,1:-1:,-1]           = Interm_Array_1[1:-1:,1:-1:,-2]
            
            return Interm_Array_1
        
        return DouglasGunnLoop
    
    def get_run_direction_x(self):
        run_matbuilder = self.run_matbuilder
        run_thomas = self.run_thomas
        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL, parallel=self.PARALLEL)
        def run_direction_x(START, stack_one_Third, ll, Interm_Array_1):
            for j in prange(len(START[0,::,0])):
                for k in range(len(START[0,0,::])):
                    aa,bb,cc                          = run_matbuilder(START[::,j,k], ll)
                    stack_one_Third[1:-1:,j+1,k+1]    = run_thomas(aa,bb,cc, Interm_Array_1[1:-1:,j+1,k+1])
            return stack_one_Third

        return run_direction_x
    
    def get_run_direction_y(self):
        run_matbuilder = self.run_matbuilder
        run_thomas = self.run_thomas
        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL, parallel=self.PARALLEL)
        def run_direction_y(START, stack_two_Third, ll, Interm_Array_1):
            for i in range(len(START[::,0,0])):
                for k in range(len(START[0,0,::])):
                    aa,bb,cc                          = run_matbuilder(START[i,::,k], ll)
                    stack_two_Third[i+1,1:-1:,k+1]    = run_thomas(aa,bb,cc, Interm_Array_1[i+1,1:-1:,k+1])
            return stack_two_Third

        return run_direction_y
    
    def get_run_direction_z(self):
        run_matbuilder_z = self.run_matbuilder_z
        run_thomas = self.run_thomas
        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL, parallel=self.PARALLEL)
        def douglas_loop_z(START, Interm_Array_1, Universal_LD_z, Universal_MD_z, Universal_UD_z, DiffKoeff, TimeIncr):
            for i in prange(len(START[::,0,0])):
                for j in prange(len(START[0,::,0])):
                    aa,bb,cc                           = run_matbuilder_z(START[i,j,::], Universal_LD_z, Universal_MD_z, Universal_UD_z, DiffKoeff, time_incr = TimeIncr)
                    Interm_Array_1[i+1,j+1,1:-1:]      = run_thomas(aa,bb,cc, Interm_Array_1[i+1,j+1,1:-1:])
            return Interm_Array_1
        
        return douglas_loop_z
    
    def get_run_matbuilder(self):
        dtype = self.DTYPE

        #==========================================================================================================================
        #This function is the matrixbuilder for the Douglas-Gunn algorithm. It is basically a 1-D Crank Nicolson type of matrix
        #which involves any internal boundary conditions from a Starting-Array. It returns three vectors which will be used by the Thomas 
        #function to solve for the next time-(sub)-iteration. This function can (in this script which has an expanding z-grid)
        #only be used for the x,y-dimensions
        #==========================================================================================================================
        
        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL)
        def Matbuilder(Array_Vector, ll):
            lenx       = len(Array_Vector)
            Middle     = (2+2*ll)*np.ones(lenx, dtype=dtype)
            Middle[0]  = (2+ll)
            Middle[-1] = (2+ll)
            OffDiag    = -ll*np.ones(lenx-1, dtype=dtype)
            Middle     = Middle*Array_Vector + (Array_Vector-1)**2
            up1        = OffDiag*Array_Vector[0:-1:]
            low1       = OffDiag*Array_Vector[1::]
            return low1,  Middle, up1

        return Matbuilder

    def get_run_matbuilder_z(self):
        #==========================================================================================================================
        #This function is the matrixbuilder for the Douglas-Gunn algorithm in the z-dimension
        #==========================================================================================================================
        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL)
        def Matbuilder_Z(Array_Vector, GENERAL_LD, GENERAL_MD, GENERAL_UD, DiffKoeff, time_incr):
            LD_now       = DiffKoeff*time_incr*GENERAL_LD*Array_Vector[1::] 
            MD_now       = (2 + DiffKoeff*time_incr*GENERAL_MD)*Array_Vector + (Array_Vector - 1)**2
            UD_now       = DiffKoeff*time_incr*GENERAL_UD*Array_Vector[:-1:] 
            return LD_now, MD_now, UD_now

        return Matbuilder_Z

    def get_run_thomas(self):
        dtype = self.DTYPE
        #==========================================================================================================================
        #This is a TDMA solver aka the Thomas algorithm.
        #==========================================================================================================================
        @jit(nopython=self.NOPYTHON, nogil=self.NOGIL)
        def Thomas(a,b,c,L):
            cs = np.zeros(len(c), dtype=dtype)
            Ls = np.zeros(len(L), dtype=dtype)
            x  = np.zeros(len(L), dtype=dtype)
            cs[0] = c[0]/b[0]
            Ls[0] = L[0]/b[0]
            for i in range(len(L)-1):
                if i>0:
                    cs[i]= c[i]/(b[i] - a[i-1]*cs[i-1])
                    Ls[i]= (L[i] - a[i-1]*Ls[i-1])/(b[i] - a[i-1]*cs[i-1])
            Ls[-1] = (L[-1] - a[-1]*Ls[-2])/(b[-1] - a[-1]*cs[-1])
            x[-1] = Ls[-1]
            for i in range(len(L)):
                if i > 0:
                    x[-(i+1)] = Ls[-(i+1)] - cs[-(i)]*x[-(i)]
            return x
        
        return Thomas