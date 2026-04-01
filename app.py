import streamlit as st
import numpy as np
import signal
import pyvista as pv
import pandas as pd

pv.start_xvfb()

# --- GMSH THREADING PATCH ---
if not hasattr(signal, "original_signal"):
    signal.original_signal = signal.signal
    def safe_signal(signum, handler):
        try: return signal.original_signal(signum, handler)
        except ValueError: return None
    signal.signal = safe_signal

import gmsh

st.set_page_config(page_title="FEM Visualizer", layout="wide")
st.title("🏗️ FEM Visualizer: Multi-Hole Stress Analysis")

def run_simulation(L, H, holes_data, thick, force, E, uts, mesh_quality):
    try:
        gmsh.initialize()
        gmsh.model.add("MultiHolePlate")
        
        # Geometry: Create Base Plate
        rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        
        # Geometry: Create Multiple Holes
        disk_tags = []
        for hx, hy, hr in holes_data:
            disk = gmsh.model.occ.addDisk(hx, hy, 0, hr, hr)
            disk_tags.append((2, disk))
            
        # Boolean Cut
        gmsh.model.occ.cut([(2, rect)], disk_tags)
        gmsh.model.occ.synchronize()
        
        # Mesh Refinement
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_quality)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_quality / 5)
        gmsh.model.mesh.generate(2)
        gmsh.write("refined_mesh.msh")
        
        grid = pv.read("refined_mesh.msh")
        
        # --- PHYSICS CALCULATIONS (Multi-Hole Surrogate) ---
        max_hole_dia = max([2*hr for _, _, hr in holes_data]) if holes_data else 0
        net_area = (H - max_hole_dia) * thick
        nominal_stress = force / net_area if net_area > 0 else force / 1.0
        
        stress_field = np.ones(grid.n_points) * nominal_stress
        
        for hx, hy, hr in holes_data:
            dist = np.linalg.norm(grid.points - [hx, hy, 0], axis=1)
            dist[dist < hr] = hr 
            
            local_stress = nominal_stress * (1 + 0.5 * (hr/dist)**2 + 1.5 * (hr/dist)**4)
            stress_field = np.maximum(stress_field, local_stress)
            
        grid["Stress (MPa)"] = stress_field
        max_stress = np.max(stress_field)
        
        avg_area = ((H * thick) + net_area) / 2
        max_def = (force * L) / (E * avg_area)
        grid["Deformation (mm)"] = (stress_field / E) * L * 0.05 

        # Plotting
        p = pv.Plotter(off_screen=True, window_size=[1000, 600])
        p.add_mesh(grid, scalars="Stress (MPa)", cmap="jet", smooth_shading=True, show_scalar_bar=True)
        p.view_xy()
        p.background_color = "white"
        
        return p.screenshot(), max_stress, max_def
        
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()

# --- UI LAYOUT ---
col_in, col_viz = st.columns([1, 2])

with col_in:
    st.header("⚙️ Design Parameters")
    with st.expander("Plate Dimensions", expanded=True):
        L = st.slider("Plate Length (L)", 50, 300, 150)
        H = st.slider("Plate Height (H)", 50, 300, 100)
        th = st.number_input("Thickness (mm)", 1.0, 50.0, 5.0)
        
    with st.expander("Hole Configurations", expanded=True):
        num_holes = st.number_input("Number of Holes", 1, 5, 2)
        holes_list = []
        for i in range(int(num_holes)):
            st.markdown(f"**Hole {i+1}**")
            c1, c2, c3 = st.columns(3)
            hx = c1.number_input("X Pos", 0.0, float(L), float(L)/2 + (i*30) - 15, key=f"x{i}")
            hy = c2.number_input("Y Pos", 0.0, float(H), float(H)/2, key=f"y{i}")
            hr = c3.number_input("Radius", 1.0, 30.0, 10.0, key=f"r{i}")
            holes_list.append((hx, hy, hr))
    
    with st.expander("Material & Loading"):
        mat = st.selectbox("Material", ["Steel", "Aluminum", "Titanium"])
        props = {"Steel": (210000, 400), "Aluminum": (70000, 250), "Titanium": (110000, 900)}
        E_mod, uts_val = props[mat]
        force_n = st.number_input("Applied Axial Force (N)", 100, 100000, 10000)
    
    mesh_q = st.select_slider("Mesh Quality", options=[10.0, 5.0, 2.5, 1.0], value=5.0)

if st.button("🚀 Run Multi-Hole Analysis"):
    img, s_max, d_max = run_simulation(L, H, holes_list, th, force_n, E_mod, uts_val, mesh_q)
    
    with col_viz:
        st.subheader("📊 Key Performance Indicators")
        m1, m2, m3 = st.columns(3)
        m1.metric("Global Max Stress", f"{s_max:.2f} MPa")
        m2.metric("Max Deformation", f"{d_max:.4e} mm")
        m3.metric("UTS ({})".format(mat), f"{uts_val} MPa")

        fos = uts_val / s_max
        if fos < 1.0:
            st.error(f"🚨 FAILURE: Factor of Safety is {fos:.2f}.")
        elif fos < 2.0:
            st.warning(f"⚠️ MARGINAL: Factor of Safety is {fos:.2f}.")
        else:
            st.success(f"✅ SAFE: Factor of Safety is {fos:.2f}.")

        st.image(img, use_container_width=True)
