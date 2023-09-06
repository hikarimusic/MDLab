from model import *
import glm
from random import random, randint
import numpy as np
from scipy.stats import uniform, maxwell
import torch

class Simulate:
    def __init__(self):
        self.initial()
    
    def initial(self):
        M = 1.66e-27
        L = 1e-9 
        T = 1e-12
        # H2O
        self.n = 1000
        self.m = 18
        self.r = 0.14
        self.R = ((self.m*M)/(997*(L**3)))**(1./3.)/2
        self.L =  self.n**(1./3.)*self.R
        self.T = 310
        self.S = 0.1

        k = 1.38e-23 * (T**2)/(L**2*M)
        na = 6.02e23
        self.p = (self.m*k*self.T)**(0.5)
        self.e = (1/4)*(4.18/(na/self.m)) * (T**2)/(L**2*M)

        qi = [uniform.rvs(loc=-1, scale=2, size=self.n) * self.L,
              uniform.rvs(loc=-1, scale=2, size=self.n) * self.L,
              uniform.rvs(loc=-1, scale=2, size=self.n) * self.L]
        vec = np.random.randn(self.n, 3)
        vec /= np.expand_dims(np.sqrt(np.sum(vec*vec+1e-30, axis=1)), axis=1)
        pi = [maxwell.rvs(size=self.n) * vec[:,0] * self.p,
              maxwell.rvs(size=self.n) * vec[:,1] * self.p,
              maxwell.rvs(size=self.n) * vec[:,2] * self.p]
        self.qi = torch.tensor(np.transpose(qi), device='cuda:0', requires_grad=True)
        self.pi = torch.tensor(np.transpose(pi), device='cuda:0', requires_grad=True)
        self.rp = torch.tensor(np.transpose(pi), device='cuda:0')

        self.time = 0.
        self.clip = 10

    def step(self):
        print('Time:', round(self.time, 1), 'ps          ', end='\r')
        self.time += self.S
        n, L, S, clip = self.n, self.L, self.S, self.clip
        ham = self.hamiltonian()
        ham.backward()
        with torch.no_grad():
            self.qi += torch.tanh(self.pi.grad/clip)*clip * S
            self.pi -= torch.tanh(self.qi.grad/clip)*clip * S
            self.pi[torch.where(self.qi<-L)] = self.rp[torch.randint(0, n, (n,))][torch.where(self.qi<-L)]
            self.pi[torch.where(self.qi<-L)] = torch.abs(self.pi[torch.where(self.qi<-L)])
            self.pi[torch.where(self.qi>L)] = self.rp[torch.randint(0, n, (n,))][torch.where(self.qi>L)]
            self.pi[torch.where(self.qi>L)] = -torch.abs(self.pi[torch.where(self.qi>L)])
        if self.qi.grad!=None:
            self.qi.grad.zero_()
        if self.pi.grad!=None:
            self.pi.grad.zero_()

    def hamiltonian(self):
        n, m, r, e, qi, pi = self.n, self.m, self.r, self.e, self.qi, self.pi   
        T = torch.sum((pi*pi)/(2*m))
        qd = qi[:,None,:].expand(n,n,3) - qi[None,:,:].expand(n,n,3)
        q2 = torch.sum(qd*qd, axis=2) + torch.eye(n, device='cuda:0')
        rq2 = (r**2) / q2
        rq6 = torch.pow(rq2, 3)
        rq12 = torch.pow(rq6, 2)
        V = 4 * e * (rq12-rq6) * (torch.ones(n,n, device='cuda:0')-torch.eye(n, device='cuda:0'))
        V = torch.sum(V) / 2
        H = T - V
        return H


class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.skybox = AdvancedSkyBox(app, tex_id='black')
        self.simulate = Simulate()
        self.load()

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object
        sim= self.simulate

        self.sphere = [None for i in range(sim.n)]
        for i in range(sim.n):
            self.sphere[i] = MovingSphere(app, pos=(sim.qi[i,0].item(), sim.qi[i,1].item(), sim.qi[i,2].item()), 
                                          scale=(sim.r, sim.r, sim.r), tex_id='red')
            add(self.sphere[i])

    def update(self):
        sim = self.simulate
        sim.step()
        for i in range(sim.n):
            self.sphere[i].pos = (sim.qi[i,0].item(), sim.qi[i,1].item(), sim.qi[i,2].item())
