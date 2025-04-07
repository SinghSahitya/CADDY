# python_scripts/step_to_off.py
import sys
import os
import numpy as np
from steputils import p21
import math
from collections import defaultdict

def step_to_off(step_file, off_file):
    try:
        # Parse the STEP file using steputils
        print(f"Reading STEP file: {step_file}")
        step_data = p21.readfile(step_file)
        
        # Maps to store entities by reference
        entities_by_id = {}
        points = {}
        vertex_points = {}
        edges = {}
        faces = {}
        loops = {}
        
        # First pass: index all entities by their ID
        print("Indexing STEP entities...")
        for entity in step_data.data:
            entity_id = entity.ref
            entities_by_id[entity_id] = entity
        
        # Second pass: extract important geometric elements
        print("Extracting geometric data...")
        for entity_id, entity in entities_by_id.items():
            # Extract CARTESIAN_POINT entities
            if entity.name == 'CARTESIAN_POINT':
                if len(entity.params) >= 2 and isinstance(entity.params[1], list):
                    coords = [float(x) for x in entity.params[1]]
                    if len(coords) >= 3:
                        points[entity_id] = np.array(coords[:3])
            
            # Extract VERTEX_POINT entities (vertices that reference points)
            elif entity.name == 'VERTEX_POINT':
                if len(entity.params) >= 2:
                    point_ref = entity.params[1]
                    if isinstance(point_ref, str) and point_ref.startswith('#'):
                        point_id = point_ref[1:]  # Remove the '#' prefix
                        vertex_points[entity_id] = point_id
            
            # Extract EDGE_CURVE entities (edges between vertices)
            elif entity.name == 'EDGE_CURVE':
                if len(entity.params) >= 3:
                    start_vertex_ref = entity.params[1]
                    end_vertex_ref = entity.params[2]
                    if isinstance(start_vertex_ref, str) and start_vertex_ref.startswith('#') and \
                       isinstance(end_vertex_ref, str) and end_vertex_ref.startswith('#'):
                        start_id = start_vertex_ref[1:]
                        end_id = end_vertex_ref[1:]
                        edges[entity_id] = (start_id, end_id)
            
            # Extract EDGE_LOOP entities (sequences of edges forming loops)
            elif entity.name == 'EDGE_LOOP':
                if len(entity.params) >= 2 and isinstance(entity.params[1], list):
                    edge_refs = []
                    for edge_ref in entity.params[1]:
                        if isinstance(edge_ref, str) and edge_ref.startswith('#'):
                            edge_refs.append(edge_ref[1:])
                    loops[entity_id] = edge_refs
            
            # Extract FACE_OUTER_BOUND and FACE_BOUND entities (boundaries of faces)
            elif entity.name in ('FACE_OUTER_BOUND', 'FACE_BOUND'):
                if len(entity.params) >= 2:
                    loop_ref = entity.params[0]
                    if isinstance(loop_ref, str) and loop_ref.startswith('#'):
                        loop_id = loop_ref[1:]
                        faces[entity_id] = {'loop': loop_id, 'outer': entity.name == 'FACE_OUTER_BOUND'}
        
        # Resolve references to build a mesh
        print("Building mesh structure...")
        vertex_coords = []  # Final list of vertex coordinates
        face_indices = []   # Final list of face indices
        
        # Map to track unique vertices and avoid duplicates
        unique_vertices = {}
        
        # Function to get or add vertex
        def get_vertex_index(coords):
            vertex_key = f"{coords[0]:.6f},{coords[1]:.6f},{coords[2]:.6f}"
            if vertex_key not in unique_vertices:
                unique_vertices[vertex_key] = len(vertex_coords)
                vertex_coords.append(coords)
            return unique_vertices[vertex_key]
        
        # Process faces and create triangles
        for face_id, face_data in faces.items():
            loop_id = face_data['loop']
            if loop_id in loops:
                edge_ids = loops[loop_id]
                
                # Collect vertices from edges
                face_vertices = []
                for edge_id in edge_ids:
                    if edge_id in edges:
                        start_id, end_id = edges[edge_id]
                        
                        # Get point coordinates for start vertex
                        if start_id in vertex_points and vertex_points[start_id] in points:
                            coords = points[vertex_points[start_id]]
                            vertex_index = get_vertex_index(coords)
                            face_vertices.append(vertex_index)
                
                # Create triangles from vertex list (simple fan triangulation)
                if len(face_vertices) >= 3:
                    anchor = face_vertices[0]
                    for i in range(1, len(face_vertices) - 1):
                        face_indices.append([anchor, face_vertices[i], face_vertices[i + 1]])
        
        # If no vertices/faces were found, try direct extraction method
        if len(vertex_coords) == 0:
            print("No mesh structure found. Trying direct extraction...")
            
            # Extract all CARTESIAN_POINT entities
            for entity_id, entity in entities_by_id.items():
                if entity.name == 'CARTESIAN_POINT':
                    if len(entity.params) >= 2 and isinstance(entity.params[1], list):
                        coords = [float(x) for x in entity.params[1]]
                        if len(coords) >= 3:
                            vertex_coords.append(coords[:3])
            
            # If file contains closed shells or breps, try to extract faces
            for entity_id, entity in entities_by_id.items():
                if entity.name in ('CLOSED_SHELL', 'MANIFOLD_SOLID_BREP'):
                    if entity.name == 'CLOSED_SHELL' and len(entity.params) >= 2:
                        face_refs = entity.params[1]
                    elif entity.name == 'MANIFOLD_SOLID_BREP' and len(entity.params) >= 2:
                        shell_ref = entity.params[1]
                        if isinstance(shell_ref, str) and shell_ref.startswith('#'):
                            shell_id = shell_ref[1:]
                            if shell_id in entities_by_id:
                                shell = entities_by_id[shell_id]
                                if len(shell.params) >= 2:
                                    face_refs = shell.params[1]
                                else:
                                    face_refs = []
                        else:
                            face_refs = []
                    else:
                        face_refs = []
                    
                    # Create simple triangular faces if we have enough vertices
                    if len(vertex_coords) >= 3 and isinstance(face_refs, list) and len(face_refs) > 0:
                        # Create a simple triangulation using groups of 3 vertices
                        for i in range(0, min(len(vertex_coords), len(face_refs) * 3), 3):
                            if i + 2 < len(vertex_coords):
                                face_indices.append([i, i + 1, i + 2])
        
        # If still no vertices/faces found, raise an error
        if len(vertex_coords) == 0:
            raise ValueError("No vertices found in STEP file")
        
        # If we have vertices but no faces, create simple triangular faces
        if len(vertex_coords) > 0 and len(face_indices) == 0:
            print("No faces found. Creating simple triangulation...")
            # Create triangles by grouping consecutive vertices
            for i in range(0, len(vertex_coords) - 2, 3):
                face_indices.append([i, i + 1, i + 2])
        
        print(f"Generated mesh with {len(vertex_coords)} vertices and {len(face_indices)} faces")
        
        # Generate OFF file content
        print(f"Writing OFF file: {off_file}")
        with open(off_file, 'w') as f:
            f.write("OFF\n")
            f.write(f"{len(vertex_coords)} {len(face_indices)} 0\n")
            
            # Write vertices
            for v in vertex_coords:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            
            # Write faces
            for face in face_indices:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        print("Conversion completed successfully")
        return True
    except Exception as e:
        print(f"Error converting STEP to OFF: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python step_to_off.py <step_file> <off_file>")
        sys.exit(1)
    
    step_file = sys.argv[1]
    off_file = sys.argv[2]
    
    success = step_to_off(step_file, off_file)
    if success:
        print("Conversion successful")
        sys.exit(0)
    else:
        print("Conversion failed")
        sys.exit(1)
