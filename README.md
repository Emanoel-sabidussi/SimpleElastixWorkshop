## Option-1 
Run the notebook on Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Emanoel-sabidussi/SimpleElastixWorkshop/master)



## Option-2 
Use Docker! 

Warning: Pulled image (3.5GB) expands to 14GB, be careful about this option if your disk space is low. 

* Pull the Docker image 

`docker pull agahkarakuzu/elastixmrb` 

The `Dockerfile` used to build `agahkarakuzu/elastixmrb` is available at `Docker` folder. 

* Run the Docker image

`docker run -it --rm -p 8888:8888 agahkarakuzu/elastixmrb` 

* Go to your web browser and visit `localhost:8888`, which will be asking for a password. This password is the hash displayed on the terminal. Copy the hash that follows `?token=` expression, and paste it to the password section on your browser.  

`http://127.0.0.1:8888/?token=b47gjhjrt465tuewadhy4tw546fsjtwy`

## Option-3 

Hi! To (locally) run the hands-on session of the SimpleElastix workshop at the OpenMR Benelux 2020,
there are a couple of requirements that should be installed previous to the start of the session:

- Jupyter Notebook
- Python>3
- torch 1.3
- SimpleElastix - Python binding
- nibabel
- h5py
- scipy
- matplotlib
- ipywidgets

If you are having any problems installing any of the packages, especially SimpleElastix, 
which have some known issues with MacOS, please contact me via our Slack channel, or DM me there,
I will be happy to give you some support!

Slack: OpenMR2020 - [Workshop Help Center](https://openmrworkspace.slack.com/archives/CSGTF3L4R)

Cheers!



