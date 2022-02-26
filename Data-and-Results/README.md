# Base-Problems/
This directory contains example datasets for the DMC-Topology-Fix in VisualDMC.
The isovalue triggering these cases is 900 (except for files that contain the iso-value in their name).
All Files should be loaded without the 3-Fix (which extends the Grid by 1 in each direction).

- Files with the name "n-Edge.bin" contain at least one quad with "n" non-manifold edges.
- "Iterative-Fix.bin" contains a minimum example where at least two iterations are necessary to fix non-manifold edges.
- "Non-Manifold-Vertex.bin" contains an example that creates a non-manifold vertex at the boundaries of the grid.
- "12-Merge-iso-241.525.bin" contains an example of a Vertex that connects 6 non-manifold edges. The Merge-Step merges 12 Vertices in this case. The case of interest is produced at an iso-value of 241.525.

# Base-Results/
This directory contains Mesh-Results of the datasets from Base-Problems.
The results are divided by enabled post-process fixes.