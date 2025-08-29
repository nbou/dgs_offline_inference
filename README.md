## Code to run Deployment inference on data collected by ReefScan/ASV

---

### Prerequisites

1. Install docker compose
```bash
sudo apt-get install docker-compose
```
2. Allow X11 connections to local docker containers so the file dialogue displays work
```bash
xhost +
```
---

### Build and Run
All these commands should be run from the root of the repository.

To build the container run:
```bash
docker compose build
```
To run the container run:
```bash
docker compose up
```

To rerun the container after stopping it run this command again.

If you make changes to the code you will need to rebuild the container.
```bash
docker compose down
docker compose build
docker compose up
```

---
### Usage:

Once the container is running you should see a window pop up with the GUI.
If it's the first time you'll need to set the locations for the model file, and the data input and output directories.
If you make a mistake here, you'll need to remove the created config file using this command before runnign again:
```bash
rm /media/jetson/asv_data/PROCESSED_DATA/system_config.ini
````

Then you'll provide the directory containing the survey to inference. Make sure you click the top level dir of the 
survey (not the individual camera dir). After this you'll provide the camera number (1 or 2), and a value for the 
number of frames to skip between inference (e.g. 10 means every 10th frame will be inferred).
The script will generate an output directory matching the name of the survey/camera.  

---

### Alternative: Manual Docker build and run commands
If you have issues with docker compose you can build and run the container manually.

docker build command:
```bash
docker build -t dgs_offline -f docker/Dockerfile .
```

docker run command:
```bash
    docker run --runtime nvidia -it --network=host  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY dgs_offline
```
