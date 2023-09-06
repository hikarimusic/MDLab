from model import *
import glm
from random import random, randint
import numpy as np
from scipy.stats import uniform, maxwell
import torch
import pandas as pd
import seaborn as sns
import time

class Simulate:
    def __init__(self):
        self.device = 'cuda:0'
        # self.device = 'cpu'
        self.initial()
    
    def initial(self):
        M = 1.66e-27
        L = 1e-9 
        T = 1e-12
        Q = 1.60e-19
        # H2O
        self.n = 1000
        self.n1 = self.n // 2
        self.n2 = self.n - self.n1
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

        self.dp = 1.84*3.33e-30 / (Q*L)
        K = 8.99e9 * ((T**2)*(Q**2))/(M*(L**3))
        self.e2 = K*self.dp*self.dp

        qi = [uniform.rvs(loc=-1, scale=2, size=self.n) * self.L,
              uniform.rvs(loc=-1, scale=2, size=self.n) * self.L,
              uniform.rvs(loc=-1, scale=2, size=self.n) * self.L]
        vec = np.random.randn(self.n, 3)
        vec /= np.expand_dims(np.sqrt(np.sum(vec*vec+1e-30, axis=1)), axis=1)
        pi = [maxwell.rvs(size=self.n) * vec[:,0] * self.p,
              maxwell.rvs(size=self.n) * vec[:,1] * self.p,
              maxwell.rvs(size=self.n) * vec[:,2] * self.p]
        self.qi = torch.tensor(np.transpose(qi), device=self.device, requires_grad=True)
        self.pi = torch.tensor(np.transpose(pi), device=self.device, requires_grad=True)
        self.rp = torch.tensor(np.transpose(pi), device=self.device)

        self.time = 0.
        self.clip = 50

        self.data = []

    def step(self):
        if self.time<0.01:
            self.start_time = time.time()
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
        
        # for i in range(self.n1):
        #     self.data.append([self.qi[i,1].item(), round(self.time, 1), "A"])
        # for i in range(self.n1, self.n):
        #     self.data.append([self.qi[i,1].item(), round(self.time, 1), "B"])
        # if abs(self.time-20) < 0.01:
        #     df = pd.DataFrame(self.data, columns=["Position Y (nm)", "Time (ps)", "Particle"])
        #     sns.set_theme()
        #     plot = sns.relplot(data=df, kind="line", x="Time (ps)", y="Position Y (nm)", hue="Particle")
        #     plot.fig.savefig("plot.jpg")


        # if abs(self.time-1) < 0.01:
        #     print("")
        #     print(self.device, self.n, time.time()-self.start_time)
        #     print("")

    def hamiltonian(self):
        n, n1, m, r, L, e, e2, qi, pi = self.n, self.n1, self.m, self.r, self.L, self.e, self.e2, self.qi, self.pi   
        T = torch.sum((pi*pi)/(2*m))
        qd = qi[:,None,:].expand(n,n,3) - qi[None,:,:].expand(n,n,3)
        q2 = torch.sum(qd*qd, axis=2) + torch.eye(n, device=self.device)
        rq2 = (r**2) / q2
        rq6 = torch.pow(rq2, 3)
        rq12 = torch.pow(rq6, 2)
        V1 = 4 * e * (rq12-rq6) * (torch.ones(n,n, device=self.device)-torch.eye(n, device=self.device))
        V1 = torch.sum(V1) / 2
        V2 = - self.e2 / torch.pow(q2[:n1,:n1], 3/2)
        V2 *= torch.ones(n1, n1, device=self.device)-torch.eye(n1, device=self.device)
        V2 = torch.sum(V2) / 2
        wd2 = torch.pow((qi[:n1,1]+L),2)
        V3 = torch.sum(-self.e2 / wd2)
        V = V1 + V2 + V3
        H = T + V
        return H


class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.skybox = AdvancedSkyBox(app, tex_id='black')
        self.simulate = Simulate()
        self.app.camera.position = glm.vec3(0, 0, 6)
        self.app.capture = []
        self.load()

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object
        sim= self.simulate

        self.sphere = [None for i in range(sim.n)]
        for i in range(sim.n1):
            self.sphere[i] = MovingSphere(app, pos=(sim.qi[i,0].item(), sim.qi[i,1].item(), sim.qi[i,2].item()), 
                                          scale=(sim.r, sim.r, sim.r), tex_id='red')
            add(self.sphere[i])
        for i in range(sim.n1, sim.n):
            self.sphere[i] = MovingSphere(app, pos=(sim.qi[i,0].item(), sim.qi[i,1].item(), sim.qi[i,2].item()), 
                                          scale=(sim.r, sim.r, sim.r), tex_id='white')
            add(self.sphere[i])
        
        add(Cat(app, pos=(5, 0, 5), scale=(.1, .1, .1)))

    def update(self):
        sim = self.simulate
        for i in range(1):
            sim.step()
        for i in range(sim.n):
            self.sphere[i].pos = (sim.qi[i,0].item(), sim.qi[i,1].item(), sim.qi[i,2].item())
