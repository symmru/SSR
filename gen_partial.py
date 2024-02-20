import sys
import os

def main():
  sys.path.append("./meshcnn")
  from mesh import export_partial_spheres
  export_partial_spheres(range(2,10), "mesh_files")

if __name__ == '__main__':
  main()
