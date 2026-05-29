from gmath import *
import numpy as np
from constants import *

default_colordict =  {'red': [0.2, 0.5, 0.5],
                          'green': [0.2, 0.5, 0.5],
                          'blue': [0.2, 0.5, 0.5]} #symbols[reflect][1]

class Ray:
    def __init__(self, start , direction  ):
        self.start = start
        self.direction = direction / np.linalg.norm(direction)

    def intersects(self, scene, origin_obj=None, shadow=False, max_t = 10e5, origin_idx = None):
        #followed https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
        #and https://cs184.eecs.berkeley.edu/sp24/lecture/9-20/ray-tracing-and-acceleration-str for muller-trumbore
        interobj = None
        min_intersect = max_t#also max render
        intersectidx = -1

        for obj in scene:
            if shadow and obj.opacity == 0:
                continue

            if type(obj) == mesh:
                polygons = obj.polygons #Array of triangle faces [[x1, y1, z1], [x2, y2 z2]]
                # print(polygons.shape) # Stored as num_triangles, 3 (Vertices per triangle) , points
                A = polygons[:, 0,:-1]# Stores every first point of every triangle
                B = polygons[:, 1,:-1]# Stores every second point of every triangle
                C = polygons[:, 2,:-1]# Stores every third point of every triangle


                #Oh dear what are we doing? E1, E2,edges. S is 
                E1 = B - A
                E2 = C - A
                S = self.start - A
                S1 = np.cross(self.direction, E2)
                S2 = np.cross(S, E1)
                S1_dot_E1 = np.einsum("ij,ij->i", S1, E1)

                for i in range(len(S1_dot_E1)):
                    if S1_dot_E1[i] == 0:
                        S1_dot_E1[i] += 10e-10 #force division 

                #Muller trumbore
                t = np.einsum("ij,ij->i", S2, E2) / S1_dot_E1
                b1 = np.einsum("ij,ij->i", S1, S)/ S1_dot_E1 #b1, b2 are barycentric coords
                b2 = np.einsum("ij,j->i", S2, self.direction)/ S1_dot_E1

                valid = (
                    (b1 >= 0) & (b2 >= 0) & #Both bs are valid
                    (b1 + b2 <= 1) & #Third barycentric coord is > 1
                    (eps <= t) & (t < min_intersect)  #t is not negative
                    ) #S1_dot_E1 was not 10e-10. Slightly hacky but should work since this would make intersection point being like rlly rlly far like out of view far

                if np.any(valid):
                    if shadow:
                        return True
                    best = np.argmin(np.where(valid, t, np.inf))
                    min_intersect = t[best]
                    interobj = obj
                    intersectidx = best

            if type(obj) == bounded_plane:
                tris = obj.tris #Array of triangle faces [[x1, y1, z1], [x2, y2 z2]]
                # print(polygons.shape) # Stored as num_triangles, 3 (Vertices per triangle) , points
                A = tris[:, 0,:-1]# Stores every first point of every triangle
                B = tris[:, 1,:-1]# Stores every second point of every triangle
                C = tris[:, 2,:-1]# Stores every third point of every triangle


                E1 = B - A
                E2 = C - A
                S = self.start - A
                S1 = np.cross(self.direction, E2)
                S2 = np.cross(S, E1)
                S1_dot_E1 = np.einsum("ij,ij->i", S1, E1)  #Basically determinant, cross is ijk and then we dot w last element of det

                for i in range(len(S1_dot_E1)):
                    if S1_dot_E1[i] == 0:
                        S1_dot_E1[i] += 10e-10 #force division 

                t = np.einsum("ij,ij->i", S2, E2) / S1_dot_E1 #Time is det w/ the dot thing again
                b1 = np.einsum("ij,ij->i", S1, S)/ S1_dot_E1 
                b2 = np.einsum("ij,j->i", S2, self.direction)/ S1_dot_E1

                valid = (
                    (b1 >= 0) & (b2 >= 0) & 
                    (b1 + b2 <= 1) & 
                    (eps <= t) & (t < min_intersect)) 
                if np.any(valid):
                    if shadow:
                        return True
                    best = np.argmin(np.where(valid, t, np.inf))
                    min_intersect = t[best]
                    interobj = obj
                    intersectidx = best

                    # print("setting color")
                    obj.set_intersect_color(intersectidx, b1, b2)

            if type(obj) == plane:
                polygons = obj.points
                A = polygons[0,:-1]
                B =polygons[1,:-1]
                C = polygons[2,:-1]

                E1 = B - A
                E2 = C - A
                S = self.start - A
                S1 = np.cross(self.direction, E2)
                S2 = np.cross(S, E1)

                S1_dot_E1 = np.dot(S1, E1)
                if(S1_dot_E1 == 0):
                    continue
                
                t = np.dot(S2, E2) / S1_dot_E1
                # print(obj, "obj and origin", origin_surface)
                if( eps <= t and  t < min_intersect): 
                    if(shadow):
                        return True
                    min_intersect = t
                    intersectidx = 1
                    interobj = obj

            if type(obj) == sphere:
                L = self.start - obj.center
                a = np.dot (self.direction, self.direction)
                b = 2 * np.dot (self.direction, L)
                c = np.dot (L, L) - (obj.radius ** 2)

                delta = b * b - 4 * a * c

                if (delta >= 0):
                    d = np.sqrt (delta)
                    t1 = (-b - d) / (2 * a)
                    t2 = (-b + d)/(2 * a)

                    t_sphere = -1
                    if t1 > min_render and t1 < min_intersect:
                        t_sphere = t1
                    elif t2 > min_render and t2 < min_intersect:
                        t_sphere = t2

                    if t_sphere != -1:
                        if (shadow and obj == origin_obj):
                            continue
                        if shadow: 
                            return True
                    
                        min_intersect = t_sphere
                        intersectidx = 0
                        interobj = obj
                
        
        if shadow:
            return False
        
        if intersectidx == -1:
            return intersectidx, None, None
        else:
            return intersectidx, interobj, min_intersect
    
    
    def get_lighting_color(self, inter_obj,light,  idx = None, inter_point=None, ambient=[50, 50, 50]):
        view = -self.direction
        
        color = inter_obj.color
        if type(inter_obj) == plane:
            n = inter_obj.normal
            if inter_obj.second_color != None:
                if (inter_point[0] // inter_obj.checker_size + inter_point[2] // inter_obj.checker_size) % 2:
                    color = inter_obj.second_color

        if type(inter_obj) == bounded_plane:
            n = inter_obj.normals[idx]
            color = inter_obj.intersect_color

        elif type(inter_obj) == mesh:
            n = inter_obj.normals[idx]
        elif type (inter_obj) == sphere:
            n = inter_point - inter_obj.center
            n = n / np.linalg.norm (n)

        a = calculate_ambient(ambient,color )
        d =  calculate_diffuse(light,color, n)
        s = calculate_specular(light,color, view, n)

        return a, d, s

class light():
    def __init__(self, location, color):
        self.location = location
        self.color = color 


class solid():
    def __init__(self, color , reflectiveness, refractive_n, opacity ):
        self.color = color
        self.reflectiveness  = reflectiveness
        self.refractive_n  = refractive_n
        self.opacity=opacity
   
class sphere(solid):
    def __init__(self, center, radius, color = default_colordict, reflectiveness = 0.5,refractive_n=1, opacity=1):
        super().__init__(color, reflectiveness, refractive_n, opacity)

        self.center = np.array (center)
        self.radius = radius
 
class plane(solid):
    def __init__(self, points, color=default_colordict , second_color=None, checker_size = 0, reflectiveness=0.5, refractive_n=1, opacity=1):
        super().__init__(None, reflectiveness, refractive_n, opacity)
        self.points = points
        A = np.array(points[0,:-1]) 
        B = np.array(points[1,:-1]) 
        C = np.array(points[2,:-1]) 
        self.normal = np.cross(B -A, C - A) 
        self.normal /= np.linalg.norm(self.normal)
        self.reflectiveness = reflectiveness
        self.color = color
        if second_color == None and checker_size > 0:
            print("Warning! Checker size > 0 but no second color set")
        if second_color != None and checker_size <= 0:
            print("Warning second color set but checker size not valid")

        self.second_color = second_color
        self.checker_size = checker_size


class mesh(solid):
    def __init__(self, polygons , color=default_colordict , reflectiveness = 0.5,refractive_n=1, opacity=1):
        super().__init__(color, reflectiveness, refractive_n, opacity)
        polygons = np.array(polygons)
        polygons = polygons.reshape(int(polygons.shape[0] / 3), 3, 4)
        self.polygons = polygons 
        A = np.array(polygons[:, 0,:-1])#.reshape(3)
        B = np.array(polygons[:, 1,:-1])#.reshape(3,1)
        C = np.array(polygons[:, 2,:-1])#.reshape(3,1)

        self.normals = np.cross(B -A, C - A) 
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1)[:, np.newaxis]
        
class bounded_plane(solid):
    def __init__(self, p0, p1, p2, p3, color=None, tmap=None, reflectiveness = 0.5,refractive_n=1, opacity=1):
        #Points go clockwise from p0.
        super().__init__(color, reflectiveness, refractive_n, opacity)
        
        self.tris = np.array([[p0, p1, p3], [p2, p3, p1]]) #upper left, upper right
        polygons = self.tris

        # polygons = polygons.reshape(int(polygons.shape[0] / 3), 3, 4)

        A = np.array(polygons[:, 0,:-1])#.reshape(3)
        B = np.array(polygons[:, 1,:-1])#.reshape(3,1)
        C = np.array(polygons[:, 2,:-1])#.reshape(3,1)

        self.normals = np.cross(B -A, C - A) 
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1)[:, np.newaxis]
        if isinstance(tmap, np.ndarray):
            self.tmap = tmap[:]

        if self.color:
            self.color =  color #{'red': color[0] * 3,'green':color[1] * 3,'blue':color[2] * 3} 
        else:
            self.color = None
    
    def set_intersect_color(self, triangle_idx, u, v):
        v, u= u[triangle_idx], v[triangle_idx]   
        #u is the horizontal axis (cols)
        if self.color:
            self.intersect_color= self.color
        else:
            if triangle_idx == 0:
                color_here = self.tmap[self.tmap.shape[0] - 1 - int(v * (self.tmap.shape[0] -1)), self.tmap.shape[1] -1 - int(u * (self.tmap.shape[1]-1))  ,:].astype(np.float32)
            else:
                color_here = self.tmap[int(v * (self.tmap.shape[0] -1)),  int(u * (self.tmap.shape[1] -1)),:].astype(np.float32)
            color_here /= 255
            self.intersect_color = {'red': [color_here[0] ,color_here[0], 0.6 * color_here[0]],
                                    'green': [color_here[1], color_here[1],0.6 * color_here[1]],
                                    'blue': [color_here[2], color_here[2],0.6 *  color_here[2]]} 
        



