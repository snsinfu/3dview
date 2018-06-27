# 3D visualizer

Vispy/opengl script I created in 2015 to visualize simple polymer model. The
script requires obsolete PyQt4 and old version of vispy, so conda is required.

## Example usage

Create a conda environment:

```console
conda env create -f environment.yml
```

Activate the environment and open an example:

```console
source activate qt4-vispy
./run example/example.json example/example.dat
```
