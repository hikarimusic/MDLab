from model import *
import glm
from random import random, randint


class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()
        # skybox
        self.skybox = AdvancedSkyBox(app, tex_id='black')

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        self.sphere = {}
        self.sphere_d = (5, 0.31, 0.28)
        (ct, sp, sz) = self.sphere_d
        for i in range(-ct, ct):
            for j in range(-ct, ct):
                for k in range(-ct, ct):  
                    rand = randint(0, 3)
                    if rand==0:
                        tex_id = 'gray'
                    elif rand==1:
                        tex_id = 'white'
                    elif rand==2:
                        tex_id = 'red'
                    elif rand==3:
                        tex_id = 'blue'
                    self.sphere[(i, j, k)] = MovingSphere(app, pos=(i*sp, j*sp, k*sp), scale=(sz/2, sz/2, sz/2), tex_id=tex_id)
                    add(self.sphere[(i, j, k)])

    def update(self):
        (ct, sp, sz) = self.sphere_d
        for i in range(-ct, ct):
            for j in range(-ct, ct):
                for k in range(-ct, ct):
                    (x, y, z) = self.sphere[(i, j, k)].pos
                    (x, y, z) = (x+(random()-0.5)*sz/2, y+(random()-0.5)*sz/2, z+(random()-0.5)*sz/2)
                    self.sphere[(i, j, k)].pos = (x, y, z)
