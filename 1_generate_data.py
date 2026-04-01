import os
import numpy as np
import gmsh
import meshio
from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from ufl import sym, grad, tr, Identity, sqrt, inner, dot, Measure, dx

NUM_SAMPLES = 20
OUTPUT_DIR = "dataset_ellipse"

def solve_plate(sample_id, rx, ry, cx, cy):
    gmsh.initialize()
    gmsh.model.add("plate")
    L, H = 1.0, 0.5
    
    plate = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
    hole = gmsh.model.occ.addDisk(cx, cy, 0, rx, ry)
    gmsh.model.occ.cut([(2, plate)], [(2, hole)])
    gmsh.model.occ.synchronize()
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.04)
    gmsh.model.mesh.generate(2)
    gmsh.write("temp.msh")
    gmsh.finalize()
    
    msh = meshio.read("temp.msh")
    points_2d = msh.points[:, :2]
    triangle_cells = [("triangle", cell.data) for cell in msh.cells if cell.type == "triangle"]
    
    out_mesh = meshio.Mesh(points=points_2d, cells=triangle_cells)
    out_mesh.write("temp.xdmf")
    
    with io.XDMFFile(MPI.COMM_WORLD, "temp.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    W = fem.functionspace(domain, ("Lagrange", 1))
    
    E, nu = 210e9, 0.3
    mu = fem.Constant(domain, E / (2 * (1 + nu)))
    lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
    
    fdim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(np.zeros(domain.geometry.dim, dtype=PETSc.ScalarType), 
                         fem.locate_dofs_topological(V, fdim, left_facets), V)
    
    right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], L))
    traction_tag = mesh.meshtags(domain, fdim, np.array(right_facets, dtype=np.int32), np.full_like(right_facets, 1, dtype=np.int32))
    ds = Measure("ds", domain=domain, subdomain_data=traction_tag)
    
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    def epsilon(u): return sym(grad(u))
    def sigma(u): return lmbda * tr(epsilon(u)) * Identity(len(u)) + 2 * mu * epsilon(u)
    
    traction = fem.Constant(domain, PETSc.ScalarType((1e7, 0.0))) 
    a = inner(sigma(u), epsilon(v)) * dx
    L_form = dot(traction, v) * ds(1)
    
    problem = LinearProblem(a, L_form, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix=f"solve_{sample_id}")
    uh = problem.solve()
    
    s = sigma(uh) - (1./3)*tr(sigma(uh))*Identity(len(uh))
    von_mises = sqrt(1.5 * inner(s, s))
    
    vm_problem = LinearProblem(inner(ufl.TrialFunction(W), ufl.TestFunction(W))*dx, inner(von_mises, ufl.TestFunction(W))*dx, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix=f"vm_{sample_id}")
    vm_field = vm_problem.solve()
    
    np.savez(f"{OUTPUT_DIR}/data_{sample_id}.npz", nodes=domain.geometry.x, stress=vm_field.x.array.real, params=np.array([rx, ry, cx, cy]))
    print(f"Sample {sample_id} saved: Rx={rx:.3f}, Ry={ry:.3f}, Pos=({cx:.2f}, {cy:.2f})")
    
    os.remove("temp.msh")
    os.remove("temp.xdmf")
    os.remove("temp.h5")

if __name__ == "__main__":
    for i in range(NUM_SAMPLES):
        rx, ry = np.random.uniform(0.02, 0.08, 2)
        cx, cy = np.random.uniform(0.3, 0.7), np.random.uniform(0.15, 0.35)
        try: solve_plate(i, rx, ry, cx, cy)
        except Exception as e: print(f"Skipped {i}: {e}")
