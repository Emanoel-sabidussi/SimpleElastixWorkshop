FROM agahkarakuzu/elastixmrb

RUN cd $HOME/work;\
    git clone https://github.com/Emanoel-sabidussi/SimpleElastixWorkshop.git --depth 1
    
WORKDIR $HOME/work

USER $NB_UID
