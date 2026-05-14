# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:03:58 2025

@author: manzoni
"""

import sys
import numpy as np 
import nemaktis as nm 
from scipy.io import loadmat
from PIL import Image
from pathlib import Path
import os

def get_img_np(viewer,polariser_angle=0,analyser_angle=90,grayscale=False):
    viewer.analyser_angle = analyser_angle;
    viewer.polariser_angle = polariser_angle;
    viewer.grayscale = grayscale;
    viewer.update_image()
    img = viewer.get_image()
    return img

def save_np_img(img,path,img_name):
    # If last dim is 1 (grayscale), remove the singleton dimension
    if img.shape[2] == 1:
        img_to_save = img[:, :, 0]
    else:
        img_to_save = img
    # Convert to uint8 if necessary (common image dtype)
    if img_to_save.dtype != np.uint8:
        img_to_save = (img_to_save * 255).astype(np.uint8)  # if img is float in [0,1]
    # Create PIL image
    if img_to_save.ndim == 2:  # grayscale
        pil_img = Image.fromarray(img_to_save, mode='L')
    else:  # color
        pil_img = Image.fromarray(img_to_save, mode='RGB')
    # Define path
    output_path = Path(path) / img_name
    # Create folder if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save image
    pil_img.save(output_path)

def transform_vtk_vti(input_vtk,output_vti,have_n=True):
    '''Insert the complete path of the vtk file and the complete path of the vti file that you want to save. 
     If you are working in the same directory you can just pass the name of the image.'''
    import numpy as np
    import pyvista as pv
    
    # 1) Load your structured grid
    src = pv.read(input_vtk)  # your .vtk file
    print(type(src), src)
    
    # Sanity check
    # assert isinstance(src, pv.StructuredGrid), "The input must be a StructuredGrid."
    
    # 2) Decide target dimensions (match the original)
    nx, ny, nz = src.dimensions  # (40, 40, 20)
    
    # 3) Build an ImageData with same bounds and matching resolution
    xmin, xmax, ymin, ymax, zmin, zmax = src.bounds
    
    # spacing so that we have nx,ny,nz points across bounds
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    dz = (zmax - zmin) / (nz - 1)
    
    img = pv.ImageData()
    img.origin     = (xmin, ymin, zmin)
    img.spacing    = (dx, dy, dz)
    img.dimensions = (nx, ny, nz)   # points along each axis
    
    # 4) If your arrays are cell-centered and you want point-centered interpolation, convert:
    # (Safe even if there's no cell data)
    src_for_sampling = src
    # src_for_sampling = src.cell_data_to_point_data(copy=False)
    
    # 5) Resample all arrays from the structured grid onto the uniform image grid
    img_resampled = img.sample(src_for_sampling)  # alias: img.resample(src_for_sampling)
    
    # 6) Save as .vti
    # save the director as a vector named "n" as required by Nemaktis
    if not have_n:
        img_resampled['n'] = img_resampled['vector_field']
    img_resampled.save(output_vti)
    
    # 7) Verify
    out = pv.read(output_vti)
    print("Output type:", type(out))
    print("Dimensions:", out.dimensions)
    print("Spacing:", out.spacing)
    print("Origin:", out.origin)
    print("Point arrays:", list(out.point_data.keys()))
    print("Cell arrays:",  list(out.cell_data.keys()))


import pyvista as pv

# mesh = pv.read("./VTK/GM_test01.vtk")
# name_director="test01" #  "Excitation02"
# name_director="Excitation02"
#name_director="Excitation02_twist"





print("almeno qui ci arriviamo?")

# name_director="Directorfield_sinxy_300_theta90v"
name_director="final_TEST_10_mae_007.4"
have_n=True
path_directors = r"C:/Users/user/Downloads/FYP/Director_simulations/Nodir/predictions_mk5/final/"
transformtovti = True 

filename = "%s%s.vtk"%(path_directors,name_director)

print(filename)
print(filename[:-3]+'vti')


if transformtovti:
    transform_vtk_vti(input_vtk=filename,output_vti=filename[:-3]+'vti',have_n=have_n)

filename = filename[:-3]+'vti'

print(filename)

mesh = pv.read(filename)

print(mesh.dimensions[0])  
print(mesh.dimensions[1])  
print(mesh.dimensions[2])  


print('ciao 0')

Nx_final = mesh.dimensions[0]
Ny_final = mesh.dimensions[1]
Nz_final = mesh.dimensions[2]


print(Nx_final,Ny_final,Nz_final)                
#
#nfield = nm.DirectorField(vti_file="./VTK/GM_test01.vtk",mesh_lengths=(13,13,3), # (Lx, Ly, Lz) microns
#                          mesh_dimensions=(Nx_final,Ny_final,Nz_final))

print(mesh)

# nfield = nm.DirectorField(vti_file="./VTK/GM_test01.vtk")

# nfield = nm.DirectorField(vti_file="./VTK/output.vti", vti_array="vector_field")

# nfield = nm.DirectorField(vti_file=filename, vti_array="vector_field",mesh_lengths=(13,13,3))
nfield = nm.DirectorField(vti_file=filename,mesh_lengths=(13,13,100))

nfield.normalize()

 
#create the set up for the liquid cristal 
mat = nm.LCMaterial(
    lc_field=nfield,ne=1.750,no=1.526,nhost=1.0003,nin=1.51,nout=1.0003)
# add 1 mm-thick glass plate
mat.add_isotropic_layer(nlayer=1.51, thickness=1000)


print('ciao 4')


# create the array of wavelength of the light 
wavelengths = np.linspace(0.4,0.6,10) #10 

# create a light propagator object
sim = nm.LightPropagator(material=mat, 
                         wavelengths=wavelengths, 
                         max_NA_objective=0.4, 
                         max_NA_condenser=0.4, 
                         N_radial_wavevectors=1)

#print(sim.material)
#sys.exit()

print('ciao 5')

# make the light propagate
output_fields=sim.propagate_fields(method="dtmm") #dtmm bpm


print('ciao 6')


#save the results of the simulation
#output_fields.save_to_vti("PN1output.vti")
# output_fields.save_to_vti(path_directors+name_director[:-3]+'_out_field.vti')
output_fields.save_to_vti(name_director[:-3]+'_out_field.vti')


print('ciao 7')

# Use Nemaktis viewer to see the output
viewer = nm.FieldViewer(output_fields)

print('ciao 8')

#viewer.plot()
img = viewer.get_image()

img = get_img_np(viewer,polariser_angle=0,analyser_angle=90,grayscale=False)

viewer.plot()

print('ciao 9')

# save_np_img(img,path=path_directors ,img_name=name_director[:-3]+'.png') #remember to change it based on the length of the extension mat=3, h5=2

# save_np_img(img,path=path_directors ,img_name=name_director+'.png') #remember to change it based on the length of the extension mat=3, h5=2



print('ciao 10')
#        except:
#            print('######################################################################')
#            print('######################################################################')
#            print('ERROR:')
#            print(i, " ", path_directors+name_director)
#            print('######################################################################')
#            print('######################################################################')    

#viewer.plot()