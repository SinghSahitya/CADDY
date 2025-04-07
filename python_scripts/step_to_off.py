import sys
import numpy as np
import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import topods_Face
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Poly import Poly_Triangulation

def step_to_off(step_file, off_file):
    # Initialize STEP reader
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)
    
    if status != 1:
        raise Exception("Error reading STEP file")
    
    # Transfer STEP to OpenCascade internal format
    step_reader.TransferRoot()
    shape = step_reader.Shape()
    
    # Mesh the shape with a deflection of 0.1
    mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    mesh.Perform()
    
    # Extract all vertices and faces
    vertices = []
    faces = []
    vertex_map = {}  # Map to track unique vertices
    
    # Explore all faces in the shape
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        location = None
        triangulation = BRep_Tool.Triangulation(face, location)
        
        if triangulation is not None:
            # Get mesh data
            tri = triangulation
            num_triangles = tri.NbTriangles()
            
            # Process all triangles in this face
            for i in range(1, num_triangles + 1):
                trian = tri.Triangle(i)
                idx1, idx2, idx3 = trian.Get()
                
                # Get vertex point for each index (1-based in OCC)
                for idx in [idx1, idx2, idx3]:
                    node = tri.Node(idx)
                    x, y, z = node.X(), node.Y(), node.Z()
                    
                    # Transform if needed
                    if location is not None:
                        transformed = location.Transformed(node)
                        x, y, z = transformed.X(), transformed.Y(), transformed.Z()
                    
                    # Create a unique key for this vertex
                    vert_key = f"{x:.6f},{y:.6f},{z:.6f}"
                    
                    if vert_key not in vertex_map:
                        vertex_map[vert_key] = len(vertices)
                        vertices.append((x, y, z))
                
                # Add face using vertex indices
                vert1_key = f"{tri.Node(idx1).X():.6f},{tri.Node(idx1).Y():.6f},{tri.Node(idx1).Z():.6f}"
                vert2_key = f"{tri.Node(idx2).X():.6f},{tri.Node(idx2).Y():.6f},{tri.Node(idx2).Z():.6f}"
                vert3_key = f"{tri.Node(idx3).X():.6f},{tri.Node(idx3).Y():.6f},{tri.Node(idx3).Z():.6f}"
                
                face_indices = [vertex_map[vert1_key], vertex_map[vert2_key], vertex_map[vert3_key]]
                faces.append(face_indices)
        
        explorer.Next()
    
    # Write to OFF file
    with open(off_file, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python step_to_off.py <step_file> <off_file>")
        sys.exit(1)
    
    step_file = sys.argv[1]
    off_file = sys.argv[2]
    
    try:
        result = step_to_off(step_file, off_file)
        if result:
            print("Conversion successful")
            sys.exit(0)
        else:
            print("Conversion failed")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
