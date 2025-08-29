docker build command:
```bash
docker build -t dgs_offline -f docker/Dockerfile .
```

docker run command:
```bash
    docker run --runtime nvidia -it --network=host  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY dgs_offline
```
