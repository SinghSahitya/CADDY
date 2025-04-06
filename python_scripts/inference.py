import torch
import numpy as np
import trimesh
import os
import argparse
import json
from model import PointNet2Classification

def load_and_preprocess_cad(file_path, num_points=1024):
    """Load and preprocess a CAD file for inference"""
    # Load the CAD file
    loaded_mesh = trimesh.load(file_path)
    
    # Check if it's a Scene object and extract the mesh
    if isinstance(loaded_mesh, trimesh.Scene):
        # Get the first geometry item from the scene
        if len(loaded_mesh.geometry) > 0:
            # Get the first mesh from the geometry dictionary
            mesh_key = list(loaded_mesh.geometry.keys())[0]
            mesh = loaded_mesh.geometry[mesh_key]
        else:
            raise ValueError("Loaded Scene object contains no geometry")
    else:
        # It's already a mesh object
        mesh = loaded_mesh
    
    # Now sample points from the mesh
    points = mesh.sample(num_points)
    
    # Continue with your existing processing...
    # Center the point cloud
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale to unit sphere
    furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / furthest_distance
    
    # Convert to tensor and return
    points_tensor = torch.from_numpy(points.astype(np.float32))
    points_tensor = points_tensor.unsqueeze(0)
    
    return points_tensor, points

def predict_cad_category(model, points_tensor, class_names, device):
    """Predict the category of a CAD file"""
    # Move tensor to device
    points_tensor = points_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(points_tensor)
        
    # Get predicted class and all probabilities
    probabilities = torch.exp(outputs)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class].item() * 100
    
    # Get top-3 predictions
    top_probs, top_indices = torch.topk(probabilities[0], 3)
    top_predictions = [{"className": class_names[idx], "probability": prob.item() * 100} 
                       for idx, prob in zip(top_indices, top_probs)]
    
    return predicted_class, confidence, top_predictions

def main():
    parser = argparse.ArgumentParser(description='Classify a single CAD file using trained PointNet++')
    parser.add_argument('--cad_file', type=str, required=True,
                       help='Path to the CAD file (.off format)')
    parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(__file__), 'checkpoints', 'pointnet_best.pth'),
                       help='Path to the trained model checkpoint')
    parser.add_argument('--num_points', type=int, default=1024,
                       help='Number of points to sample from the CAD model')
    parser.add_argument('--output_points', type=str, default='false',
                       help='Whether to include point cloud data in the output')
    
    args = parser.parse_args()
    output_points = args.output_points.lower() == 'true'

    # Check if the CAD file exists
    if not os.path.exists(args.cad_file):
        result = {"error": f"CAD file {args.cad_file} does not exist"}
        print(json.dumps(result))
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class names (ModelNet10)
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 
                  'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    
    # Load and preprocess the CAD file
    try:
        points_tensor, points = load_and_preprocess_cad(args.cad_file, num_points=args.num_points)
    except Exception as e:
        result = {"error": f"Error processing CAD file: {str(e)}"}
        print(json.dumps(result))
        return
    
    # Load the model
    try:
        model = PointNet2Classification(num_classes=len(class_names)).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        result = {"error": f"Error loading model: {str(e)}"}
        print(json.dumps(result))
        return
    
    # Predict the category
    try:
        predicted_class, confidence, top_predictions = predict_cad_category(
            model, points_tensor, class_names, device)
    except Exception as e:
        result = {"error": f"Error making prediction: {str(e)}"}
        print(json.dumps(result))
        return
    
    # Prepare the result JSON
    result = {
        "predictedClass": class_names[predicted_class],
        "confidence": confidence,
        "topPredictions": top_predictions,
        "fileName": os.path.basename(args.cad_file)
    }
    
    # Include point cloud data if requested
    if output_points:
        # Convert points to a serializable format (list of lists)
        point_list = points.tolist()
        result["pointCloud"] = point_list
    
    # Output as JSON
    print(json.dumps(result))

if __name__ == "__main__":
    main()
