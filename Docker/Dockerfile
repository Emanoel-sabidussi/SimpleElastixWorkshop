FROM jupyter/base-notebook:d4e60350af15

USER root

COPY . /tmp/
RUN apt-get update && \
    cat /tmp/apt.txt | xargs sudo apt-get install -y --no-install-recommends &&\
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
RUN cd $HOME/work; \    
    git clone https://github.com/SuperElastix/SimpleElastix;\
    mkdir build; \ 
    cd build; \
    cmake ../SimpleElastix/SuperBuild; \
    make -j6; \
    cd $HOME/work/build/SimpleITK-build/Wrapping/Python; \
    python Packaging/setup.py install; \
    cd $HOME/work; \
    git clone https://bitbucket.org/e_sabidussi/simpleelastix-workshop.git --depth 1

COPY requirements.txt /tmp 
RUN pip install --upgrade pip; \
    pip install --requirement /tmp/requirements.txt
    
WORKDIR $HOME/work

USER $NB_UID

