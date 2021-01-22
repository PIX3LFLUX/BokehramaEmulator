# BokehramaEmulator
Jeff Manderscheid
Faculty of Electrical Engineering and Information Technology, Hochschule Karlsruhe
Bokehrama Emulation V1.0, 22.09.2020

The Bokehrama Emulation can create Bokehramas out of smartphone pictures of variable sizes. The directory to a folder containing the images of the type JPG is the input.The standard input folder is the "images" folder. The output folder is the "bokehramas" folder. The masks and depth maps are saved in the folder "masks&maps".

How to install:
Create a Conda environment with Python 3.7 with the command:
conda create -n bokehrama python=3.7

The requirements necessary to run can be found in the "requirements.txt".
I recommend to install them manually in the enviroment.
OpenCV is available in conda-forge.
When pip install or conda install does not work use:
conda install -c conda-forge PACKAGENAME


The Bokehrama Emulation uses 2 Neural networks with pretrained models.

The semantic segmentation is done by the PSPNET_50 trained with ADE_20K dataset. The implementation from David Gupta
is available on Github:
https://github.com/divamgupta/image-segmentation-keras
It will be installed with the requirement.txt.
For further information take a look at the Github repository.

The depth estimation is realised by Monodepth2 which is also available on Github:
https://github.com/nianticlabs/monodepth2
"The associated Paper: Digging into Self-Supervised Monocular Depth Prediction" - by Godard, Clément; Aodha, Oisin Mac; Firman, Michael; Brostow, Gabriel
The Monodepth2 is already integrated in the Bokehrama Emulation Software, unlike the segmentation no installation is needed.
On the first run, the weights are automatically installed.


How to run:
Change the directory to the BokehramaEmulation folder.
Use python to run the main.py with standard parameters with the command:
python main.py
To set the exponent to change the depth of field, and/or set the kernel size for the Gaussian kernel for filtering by:
python main.py --exponent NUMBER --kernelsize NUMBER
The exponent needs to be greater than 2 (e.g. 3,4,5), the kernelsize must end with a 1 (e.g. 11,21,31,...)
The standard exponent is 2 and the standard kernelsize is 21
