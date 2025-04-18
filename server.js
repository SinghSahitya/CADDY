const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const stl = require('stl');
const app = express();
const port = process.env.PORT || 5000;

// Enable CORS
app.use(cors());
app.use(express.json());

// Configure file storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    // Generate a unique filename with original extension
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'cad-' + uniqueSuffix + path.extname(file.originalname));
  }
});

function convertStlToOff(stlFilePath) {
  return new Promise((resolve, reject) => {
    try {
      // Read the STL file
      const stlData = fs.readFileSync(stlFilePath);
      
      // Parse STL file
      const mesh = stl.toObject(stlData);
      
      // Create a map to track unique vertices
      const verticesMap = new Map();
      const uniqueVertices = [];
      const faces = [];
      
      // Process each triangle in the STL mesh
      mesh.facets.forEach(facet => {
        const faceIndices = [];
        
        // Process each vertex in the triangle
        facet.verts.forEach(vert => {
          const vertKey = `${vert[0]},${vert[1]},${vert[2]}`;
          
          // Check if we've seen this vertex before
          let vertexIndex = verticesMap.get(vertKey);
          
          if (vertexIndex === undefined) {
            // New vertex - add it to our list
            vertexIndex = uniqueVertices.length;
            verticesMap.set(vertKey, vertexIndex);
            uniqueVertices.push([vert[0], vert[1], vert[2]]);
          }
          
          faceIndices.push(vertexIndex);
        });
        
        faces.push(faceIndices);
      });
      
      // Generate OFF file content
      let offData = 'OFF\n';
      offData += `${uniqueVertices.length} ${faces.length} 0\n`;
      
      // Add vertices
      uniqueVertices.forEach(vertex => {
        offData += `${vertex[0]} ${vertex[1]} ${vertex[2]}\n`;
      });
      
      // Add faces
      faces.forEach(face => {
        offData += `3 ${face[0]} ${face[1]} ${face[2]}\n`;
      });
      
      // Create output filename
      const offFilePath = stlFilePath.replace('.stl', '.off');
      
      // Write OFF file
      fs.writeFileSync(offFilePath, offData);
      
      resolve(offFilePath);
    } catch (error) {
      reject(error);
    }
  });
}

// Enhanced STEP to OFF conversion function with better logging
function convertStepToOff(stepFilePath) {
  return new Promise((resolve, reject) => {
    try {
      // Create output filename
      const offFilePath = stepFilePath.replace(/\.(step|stp)$/i, '.off');
      
      console.log(`Starting STEP to OFF conversion: ${stepFilePath} -> ${offFilePath}`);
      
      // Call the Python converter script
      const pythonProcess = spawn('python', [
        path.join(__dirname, 'python_scripts/step_to_off.py'),
        stepFilePath,
        offFilePath
      ]);
      
      let outputData = '';
      let errorData = '';
      
      pythonProcess.stdout.on('data', (data) => {
        outputData += data.toString();
        console.log(`Python Converter: ${data.toString().trim()}`);
      });
      
      pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
        console.error(`Python Converter Error: ${data.toString().trim()}`);
      });
      
      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error(`STEP conversion failed with code ${code}`);
          reject(new Error(`STEP to OFF conversion failed: ${errorData || outputData}`));
          return;
        }
        
        // Verify the OFF file was created
        if (!fs.existsSync(offFilePath)) {
          reject(new Error('STEP to OFF conversion failed: Output file was not created'));
          return;
        }
        
        console.log(`STEP to OFF conversion completed successfully: ${offFilePath}`);
        resolve(offFilePath);
      });
    } catch (error) {
      console.error('Error in convertStepToOff:', error);
      reject(error);
    }
  });
}

const fileFilter = (req, file, cb) => {
  const fileName = file.originalname.toLowerCase();
  if (fileName.endsWith('.off') || fileName.endsWith('.stl') || 
      fileName.endsWith('.step') || fileName.endsWith('.stp')) {
    cb(null, true);
  } else {
    cb(new Error('Only .OFF, .STL, and .STEP files are allowed'), false);
  }
};


const upload = multer({ 
  storage: storage,
  fileFilter: fileFilter
});
app.post('/api/classify', upload.single('cadFile'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No CAD file uploaded' });
  }

  let filePath = req.file.path;
  const originalFilename = req.file.originalname.toLowerCase();
  let convertedFile = null;
  
  try {
    // Check file type and convert if needed
    if (originalFilename.endsWith('.stl')) {
      console.log(`Converting STL file to OFF format: ${filePath}`);
      filePath = await convertStlToOff(filePath);
      convertedFile = filePath; // Track converted file for cleanup
    } else if (originalFilename.endsWith('.step') || originalFilename.endsWith('.stp')) {
      console.log(`Converting STEP file to OFF format: ${filePath}`);
      filePath = await convertStepToOff(filePath);
      convertedFile = filePath; // Track converted file for cleanup
    }
    
    console.log(`Processing file: ${filePath}`);

    // Run the Python inference script
    const pythonProcess = spawn('python', [
      path.join(__dirname, 'python_scripts/inference.py'),
      '--cad_file', filePath,
      '--output_points', 'true'
    ]);

    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      // Clean up the uploaded file and converted file if any
      fs.unlink(req.file.path, (err) => {
        if (err) console.error(`Error deleting original file: ${err}`);
      });
      
      if (convertedFile && convertedFile !== req.file.path) {
        fs.unlink(convertedFile, (err) => {
          if (err) console.error(`Error deleting converted file: ${err}`);
        });
      }

      if (code !== 0) {
        return res.status(500).json({ 
          error: 'Error processing CAD file', 
          details: errorData 
        });
      }

      try {
        // Parse the JSON output from Python
        const results = JSON.parse(outputData);
        console.log(`Classification results: ${results.predictedClass} (confidence: ${results.confidence}%)`);
        res.json(results);
      } catch (error) {
        console.error('Error parsing Python output:', error);
        res.status(500).json({ 
          error: 'Error parsing classification results',
          rawOutput: outputData
        });
      }
    });
  } catch (error) {
    // Clean up files on error
    if (req.file.path) {
      fs.unlink(req.file.path, () => {});
    }
    if (convertedFile && convertedFile !== req.file.path) {
      fs.unlink(convertedFile, () => {});
    }
    
    console.error('Error processing file:', error);
    res.status(500).json({ 
      error: 'Error processing file', 
      details: error.message 
    });
  }
});

// Add this to your server.js file
app.get('/', (req, res) => {
  res.send('CAD Classification API is running. Use POST /api/classify to classify CAD files.');
});


app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
