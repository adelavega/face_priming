FROM bamos/openface

WORKDIR /work
RUN pip install --ignore-installed  jupyter jupyterlab tornado==4.5.3
RUN pip install pandas dlib
EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
