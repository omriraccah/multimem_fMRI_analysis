import os
# Set these BEFORE any other imports
os.environ['VTK_USE_OFFSCREEN'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software rendering
os.environ['DISPLAY'] = ':99'  # Use virtual display

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from surfplot import Plot
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
from neuromaps.datasets import fetch_fslr

if __name__ == "__main__":
    surfaces = fetch_fslr(density='4k')
    lh, rh = surfaces['inflated']
    p = Plot(surf_lh=lh, surf_rh=rh)
    fig = p.build()
    fig.savefig('brain_plot.png', dpi=300, bbox_inches='tight')