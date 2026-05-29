import numpy as np
from matrix import *
from meshes import *
from objects import *
from gmath import *
from PIL import Image
import numpy as np
from constants import *
import open3d as o3d

ray_log = {}

viewerpos = np.array([0, 0, 600])
camera_z = 300
lightpos  = np.array([0,150,800]) #np.array([0, 300, 500])

def recursive_raytrace(ray, scene, lightpos, light_color, recursion_number, prev_obj=None, prev_idx=None,ray_type="source",debug_log=None):
    if recursion_number >= MAX_RECURSION:
        return BACKGROUND_COLOR#np.array([50,50,0])
    inter_idx, inter_obj, inter_t = ray.intersects(scene, origin_obj=prev_obj, origin_idx = prev_idx)

    if debug_log != None:
        ray_log[debug_log].append({"p0" : ray.start, "dir" : ray.direction, "t": inter_t, "type" : ray_type})

    if(inter_idx != -1):
        intersect_point = inter_t * ray.direction + ray.start
        light_direction = lightpos - intersect_point
        light_direction = light_direction / np.linalg.norm(light_direction)

        if (type(inter_obj) == plane ):
            inter_n = inter_obj.normal
        elif(type(inter_obj) == mesh or type(inter_obj) ==bounded_plane):
            inter_n  = inter_obj.normals[inter_idx]
        elif (type (inter_obj) == sphere):
            inter_n = intersect_point - inter_obj.center
            inter_n = inter_n / np.linalg.norm (inter_n)

        if np.dot(ray.direction, inter_n) > 0:
            # print(f"negating it w ray type {ray_type}")
            actual_inter_n = -inter_n
        else:
            actual_inter_n = inter_n

        #Bounce to light source now (?)
        shadow_ray = Ray(intersect_point + actual_inter_n * eps , lightpos - intersect_point) #shadow ray direction.

        blocked = shadow_ray.intersects(scene, shadow=True, origin_obj = inter_obj, max_t= np.linalg.norm(lightpos - intersect_point))
        if debug_log != None:
            ray_log[debug_log].append({"p0" : shadow_ray.start, "dir" : shadow_ray.direction, "t": np.linalg.norm(lightpos - intersect_point ), "type" : "shadow"})

        a,d,s = ray.get_lighting_color(inter_obj, [light_direction, light_color], inter_idx, inter_point = intersect_point)  

        color = [int(a[RED]), int(a[GREEN]), int(a[BLUE])]

        # blocked = False
        # print('no shadows')
        if(not blocked):
            color[RED] += int( d[RED] + s[RED])
            color[GREEN] += int( d[GREEN] + s[GREEN])
            color[BLUE] += int( d[BLUE] + s[BLUE])

        color = np.array(color)
        limit_color(color)

        # Reflection:
        reflection_ray = make_reflection_ray(ray, actual_inter_n, intersect_point)

        refraction_ray = make_refraction_ray(REFRACTION_N_AIR, inter_obj.refractive_n, ray, inter_n, intersect_point)
        if (refraction_ray == None):
            tir_reflection_ray = make_reflection_ray(ray, -inter_n, intersect_point)
            return recursive_raytrace(tir_reflection_ray, scene, lightpos, light_color, recursion_number + 1, inter_obj) 
        
        if inter_obj.opacity == 0:
            refracted_component = recursive_raytrace(refraction_ray, scene, lightpos, light_color, recursion_number + 1, inter_obj, ray_type = "refr",debug_log=debug_log) 
        else:
            refracted_component = np.array([0,0,0])
        
        if inter_obj.reflectiveness > 0:
            reflected_component = recursive_raytrace(reflection_ray, scene, lightpos, light_color, recursion_number + 1, inter_obj, ray_type='refl',debug_log=debug_log) 
        else:
            reflected_component = np.array([0,0,0])
        
        if inter_obj.opacity == 1:
            #Opaque
            ret = inter_obj.reflectiveness  * reflected_component + ((1 - inter_obj.reflectiveness) * color).astype(np.float64) 
        else:
            #Fresnel term
            refl_constant = calc_fresnel_reflectivity(REFRACTION_N_AIR, inter_obj.refractive_n, actual_inter_n, ray)
            ret = refl_constant * reflected_component + ((1 - refl_constant)) * refracted_component

        # ret = ((1 - inter_obj.reflectiveness) * color).astype(np.float64) 
        # ret += inter_obj.reflectiveness * reflected_component
        # limit_color(ret)
        # ret = ret * inter_obj.opacity + (1 - inter_obj.opacity) * refracted_component
        limit_color(ret)
        return ret

    return BACKGROUND_COLOR #background

def make_reflection_ray(ray, n, intersect_point):
    reflected_direction = ray.direction - 2 * np.dot(ray.direction, n) * n
    reflected_direction /= np.linalg.norm(reflected_direction)
    return Ray(intersect_point + n * eps, reflected_direction)

def make_refraction_ray(n1, n2, ray,n, intersect_point):
    #n1 is air, n2 is object.
    cos_1 = np.dot(ray.direction, n)
    if cos_1 > 0:
            # same direction as normal, so  leaving the thing and entering air.
            n1, n2 = n2, n1
            n = -n
    else:
        cos_1 *= -1
    k  = 1 - ((n1/n2) **2 ) * ( 1-  cos_1  **2) 

    if(k < 0):
        #total internal ref
        return None
    
    cos_2  = math.sqrt(1 - ((n1/n2) **2 ) * ( 1-  cos_1  **2) )
    refraction_dir  =(n1/n2) * ray.direction + ((n1/n2) *  cos_1  - cos_2 )  * n
    refraction_dir /= np.linalg.norm(refraction_dir)
    refraction_ray = Ray(intersect_point + refraction_dir * eps,  refraction_dir )
    return refraction_ray

def calc_fresnel_reflectivity(n1, n2, norm, ray):
    #n1 defaults to air, n2 is intersected obj
    cos1 = np.dot(norm, ray.direction)
    if cos1 < 0:
        cos1*= -1
    else:
        n1, n2 = n2, n1 #Another case of switched normals

    sin2_squred = (n1/n2) * (n1/n2)*(1.0-cos1*cos1)
    if sin2_squred > 1:
        return 1 #TIR
    cos2 = math.sqrt(1 - sin2_squred )
    Rs = ((n2 * cos1) - (n1 * cos2)) / ((n2 * cos1) + (n1 * cos2))
    Rp = ( (n1 * cos2) - (n2 * cos1) ) / ((n2 * cos1) + (n1 * cos2))
    return (Rs * Rs + Rp * Rp)/2


def draw_polygons_raytrace( scene, fname_to_save="out.png", resolution=1, fov=300, log_px=[], single_x = None, single_y = None, AA_radius=0):
    light_color =  [255,
                255,
                255]


    ret  = np.zeros((fov, fov, 3)) 
    if single_x != None:
        px_end = np.array([single_x, single_y, camera_z])
        px_ray = Ray(viewerpos,px_end - viewerpos)
        ray_log["RAY"] = []
        color = recursive_raytrace(px_ray,scene, lightpos, light_color, 0,  debug_log="RAY")
        color = [int(color[0]), int(color[1]), int(color[2])] 
        ret[ single_y + fov//2, single_x +  fov//2]   = color
        return

    for x in range(-fov//2, fov//2, resolution):
        for y in range(-fov//2,fov//2, resolution):
                
            px_end = np.array([x, y, camera_z])
            px_ray = Ray(viewerpos,px_end - viewerpos)
            
            should_log = False
            for thing in log_px:
                if thing[0] == x and thing[1] == y:
                    should_log = True
            
            if should_log:
                ray_log[f"{x},{y}"] = []
                color = recursive_raytrace(px_ray,scene, lightpos, light_color, 0, debug_log=f"{x},{y}")
            else:
                color = recursive_raytrace(px_ray,scene, lightpos, light_color, 0)

            color = [int(color[0]), int(color[1]), int(color[2])] 

            ret[ y + fov//2, x +  fov//2]   = color 
    
    if AA_radius != 0:
        ret_AA  = np.zeros((fov//AA_radius, fov//AA_radius, 3))

        for x in range(0, fov, AA_radius):
            for y in range(0, fov, AA_radius):
                ret_AA[y//AA_radius][x//AA_radius] = np.sum(ret[y : y + AA_radius,x: x + AA_radius], axis=(0,1))

        ret_AA/=(AA_radius ** 2)
        img = Image.fromarray(ret_AA.astype(np.uint8)[::-1,:])
        img.save("AA_" + fname_to_save)             

        
    img = Image.fromarray(ret.astype(np.uint8)[::-1,:])
    img.save(fname_to_save)


c1 =  {'red': [0.2, 0.2, 0.5],
                          'green': [0.2, 0.3, 0.5],
                          'blue': [0.5, 0.1, 0.8]} 
c2 =  {'red': [0.8, 0.8, 0.8],
                          'green': [0.2, 0.3, 0.5],
                          'blue': [0.5, 0.1, 0.8]}  

c3 =  {'red': [0.8, 0.8, 0.8],
                          'green': [0.8, 0.8, 0.8],
                          'blue': [0.8, 0.8, 0.8]} 

c4 =  {'red': [0.2, 0.8, 0.2],
                          'green': [0.8, 0.4, 0.2],
                          'blue': [0.8, 0.4, 0.3]} 

def view_debug(scene):
    geometries = []

    ray_colors_map = {
        "source": [1, 1, 0], # yellow
        "refl":   [0, 1, 0], #Green
        "refr":   [1, 0.5, 1], #Blue
        "shadow" : [0, 0, 0]
    }

    for start_ray in ray_log:
        points, lines, colors = [], [], []
        idx = 0
        for ray in ray_log[start_ray]:
            t = ray["t"] if ray["t"] is not None else 500
            p0 = ray["p0"].tolist()
            p1 = (ray["p0"] + ray["dir"] * t).tolist()
            points += [p0, p1]
            lines.append([idx, idx + 1])
            colors.append(ray_colors_map.get(ray["type"], [1, 1, 1]))
            idx += 2

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(points)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)  
        geometries.append(ls)

    for obj in scene:
        if isinstance(obj, sphere):
            mesh_s = o3d.geometry.TriangleMesh.create_sphere(radius=obj.radius)
            mesh_s.translate(obj.center)
            wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_s)
            wire.paint_uniform_color([0.8, 0.2, 0.2])
            geometries.append(wire)

        elif isinstance(obj, mesh):
            verts, tris, centers = [], [], []
            v_idx = 0
            for tri in obj.polygons:
                for pt in tri:
                    verts.append(pt[:3].tolist())
                centers.append(np.einsum('ij->j', tri)[:3] /3 )
                tris.append([v_idx, v_idx+1, v_idx+2])
                v_idx += 3

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices  = o3d.utility.Vector3dVector(verts)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(tris)
            wire = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
            wire.paint_uniform_color([0.2, 0.6, 0.9])
            geometries.append(wire)

            normal_points = []
            for i in range(len(obj.normals)):
                normal_points.append(centers[i].tolist())
                endpt = (centers[i] + 5 * obj.normals[i])
                normal_points.append(endpt.tolist())
                x = o3d.geometry.TriangleMesh.create_sphere(radius=2)
                x.translate(endpt)
                x = o3d.geometry.LineSet.create_from_triangle_mesh(x)
                x.paint_uniform_color([1,0,0])
                geometries.append(x)

            normal_lines = [ [2* i, 2 * i + 1] for i in range(len(obj.normals))]
            normal_colors = [[1, 0,0] for i in range(len(obj.normals))]

            norm_ls = o3d.geometry.LineSet()
            norm_ls.points = o3d.utility.Vector3dVector(normal_points)
            norm_ls.lines  = o3d.utility.Vector2iVector(normal_lines)
            norm_ls.colors = o3d.utility.Vector3dVector(normal_colors)  
            geometries.append(norm_ls)
    
        elif isinstance(obj, bounded_plane):
                verts, tris, centers = [], [], []
                v_idx = 0
                for tri in obj.tris:
                    for pt in tri:
                        verts.append(pt[:3].tolist())
                    centers.append(np.einsum('ij->j', tri)[:3] /3 )
                    tris.append([v_idx, v_idx+1, v_idx+2])
                    v_idx += 3

                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices  = o3d.utility.Vector3dVector(verts)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(tris)
                wire = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
                wire.paint_uniform_color([0.2, 0.6, 0.9])
                geometries.append(wire)

                normal_points = []
                for i in range(len(obj.normals)):
                    normal_points.append(centers[i].tolist())
                    endpt = (centers[i] + 5 * obj.normals[i])
                    normal_points.append(endpt.tolist())
                    x = o3d.geometry.TriangleMesh.create_sphere(radius=2)
                    x.translate(endpt)
                    x = o3d.geometry.LineSet.create_from_triangle_mesh(x)
                    x.paint_uniform_color([1,0,0])
                    geometries.append(x)

                normal_lines = [ [2* i, 2 * i + 1] for i in range(len(obj.normals))]
                normal_colors = [[1, 0,0] for i in range(len(obj.normals))]

                norm_ls = o3d.geometry.LineSet()
                norm_ls.points = o3d.utility.Vector3dVector(normal_points)
                norm_ls.lines  = o3d.utility.Vector2iVector(normal_lines)
                norm_ls.colors = o3d.utility.Vector3dVector(normal_colors)  
                geometries.append(norm_ls)

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=50))
    
    viewer = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    viewer.translate(viewerpos)
    v = o3d.geometry.LineSet.create_from_triangle_mesh(viewer)
    v.paint_uniform_color([1,1 ,0.2])
    geometries.append(v)

    light = o3d.geometry.TriangleMesh.create_sphere(radius=2)
    light.translate(lightpos)
    l = o3d.geometry.LineSet.create_from_triangle_mesh(light)
    l.paint_uniform_color([1,0.2 ,1])
    geometries.append(l)

    o3d.visualization.draw_geometries(geometries)



if( __name__ == "__main__"):
    tmp = [] #Start with emtpy array
    add_box(tmp,
         -10, 0, 0,
        300, 100, 100) #Add the polygon points
    tmp  =np.array(tmp)
    r = make_rotX(1)
    r2 = make_rotY(-.2)
    s = make_scale(.7, .7,.7)
    t = make_translate(10,40,200) #Make transformation matrices (optional)
    tmp =  (t@(r @ r2)@ s @ tmp.T).T #Apply transformations
    m = mesh( polygons=tmp,color=c4, reflectiveness=.7, refractive_n=1.5, opacity=1) #Turn it into a mesh

    tmp = [] 
    add_box(tmp,
         -10, 50, 0,
        300, 100, 100)
    tmp  =np.array(tmp)
    r = make_rotX(1)
    r2 = make_rotY(2 * math.pi * 0.7)
    r3 = make_rotZ(math.pi * .3 )
    s = make_scale(.5,.5,3)
    t = make_translate(-100,-70,50)
    tmp =  (r2 @ r3 @ s @ tmp.T).T
    m2 = mesh( polygons=tmp,color=c3, reflectiveness=.7, refractive_n=1.5, opacity=1)

    steps = 30
    max_angle = 1 * math.pi
    # for theta in range(steps):
    tmp = []
    add_mesh("shinies.STL", tmp) #Loads the STL file
    tmp  =np.array(tmp)
    center = np.average(tmp, axis=0)
    t_center = make_translate(*(-center[:-1]))
    s = make_scale(10, 10, 10)
    t = make_translate(0,0,20)
    r = make_rotX(1) #make_rotX(theta / steps * max_angle)
    tmp = (s @ t @ r @ t_center @ tmp.T).T
    dia = mesh( polygons=tmp,color=c1, refractive_n=2.42, reflectiveness=.17, opacity=0)

    plane_points = np.array([[1, -150, 1, 1], [1, -150,0 ,1], [-1,-150,0,1]], dtype=np.float64)


    duck = Image.open("mesh_files/stuyvesant.jpg") #Open image file to use as simple tmap
    #Currently only the bounded plane accepts tmap
    duck = np.array(duck)[:,:,:]

    bp1 = bounded_plane([300,-300,5,1],[300,300,5,1],[-300,300,5,1],[-300,-300,5,1],reflectiveness=0, tmap=duck)
    bp2 = bounded_plane([300,-300,5,1],[300,-300,900,1],[-300,-300,900,1],[-300,-300,5,1],reflectiveness=0, tmap=duck)
    bp3 = bounded_plane([300,300,5,1],[300,300,900,1],[300,-300,900,1],[300,-300,5,1],reflectiveness=0, tmap=duck)
    bp4 = bounded_plane([300,300,5,1],[300,300,900,1],[-300,300,900,1],[-300,300,5,1],reflectiveness=0, tmap=duck)
    bp5 = bounded_plane([-300,300,5,1],[-300,300,900,1],[-300,-300,900,1],[-300,-300,5,1],reflectiveness=0, tmap=duck)
    bp6 = bounded_plane([300,-300,900,1],[300,300,900,1],[-300,300,900,1],[-300,-300,900,1],reflectiveness=0, tmap=duck)

    scene =  [bp1, bp2, bp3, bp4, bp5, bp6, m, m2] #List of objects

    #Draws the image
    draw_polygons_raytrace(scene,f"ret.png", fov=250,resolution=1)

    # view_debug(scene)
    # print(ray_log)
