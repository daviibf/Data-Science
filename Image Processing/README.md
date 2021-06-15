## The Training Turtle
### A State-of-the-Art Pose Estimation project that aims to revolutionize the Martial Arts World.

This project was done during the AI Project Management Strategy course from the Master 2 - Big/Data IA at Hetic. It was proposed in partnership with @Nawal BENSASSI and @Olivier MAILOLS, the owners of the idea. Also, from the two steps we've mapped for reaching our goal, only the first one was done due to the limited execution time span.

#### The project consists of two parts:
 - Creation of an Image Processing model for object detection: Sensei and Apprentice.
 - Once the object was detected, we should estimate Sensei's pose. [ [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) & [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) ]


We chose [YOLOv3](https://arxiv.org/abs/2004.10934) for the detection of the fighters in the video. It is a very powerfull one. For more information, check out the [documentation](https://github.com/AlexeyAB/darknet).


The standard process is to clone [YOLO's git repository](https://github.com/AlexeyAB/darknet) so that we can use its libraries and functions. However, since we are using Docker, the approach of replicating the official [Docker Hub](https://hub.docker.com/r/daisukekobayashi/darknet) repository is more interesting because we already ensure that the environment is set up.

**Remark:** The folders `/data`,`/cfg` and `/backup` must be present in the same path of the `docker_darknet.yaml`. The model uses it for saving and loading the `.weights`, and to set the config and the architecture of the model.

Some important details before starting it:
1. Install Docker and Docker-compose if you don't have it. 
2. Depending on your computer, using the GPU can greatly speed up image processing. The communication interface with NVIDIA is done via CUDA. So, in case when performing the next steps you encounter any execution problem, it will probably be due to the CUDA version installed in your machine or the GPU version. If this is the case, since we use the darknet image directly from the Docker Hub ([Check it for some details](https://hub.docker.com/r/daisukekobayashi/darknet)). I really recommend you to check its documentation where it will give you the details to choose the correct image flag. Once chosen, you go to the Dockerfile and change the name of the defined image right after `FROM`, on the 1st line.

3. [Download](https://onlinevideoconverter.pro/en5/) the desired video you wanna use as reference.

4. Inside the `/data` file, you must put your data which gonna be used to train the model. In our case, since we are dealing with video object recognition, you have to convert the video into image frames. For that you can simply use a [online free tool](https://www.onlineconverter.com/video-to-jpg).

5. Once that done, the labels must be done on the images in order to "teach" the model what it should recognize. This task can be easily done with [CVAT](https://cvat.org/). When exporting the project, you may choose `YOLO 1.1` format, extract the zip file and move it to the `/data` folder on your project's path.

6. (optional) - Download the pre-trained weights fo the model before training it for your goal. I've used the `yolov3-tiny.conv.11`.

    - for `yolov4.cfg`, `yolov4-custom.cfg` (162 MB): [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) (Google drive mirror [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) )
    - for `yolov4-tiny.cfg`, `yolov4-tiny-3l.cfg`, `yolov4-tiny-custom.cfg` (19 MB): [yolov4-tiny.conv.29](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29)  
    - for `csresnext50-panet-spp.cfg` (133 MB): [csresnext50-panet-spp.conv.112](https://drive.google.com/file/d/16yMYCLQTY_oDlCIZPfn_sab6KD3zgzGq/view?usp=sharing)
    - for `yolov3.cfg, yolov3-spp.cfg` (154 MB): [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
    - for `yolov3-tiny-prn.cfg , yolov3-tiny.cfg` (6 MB): [yolov3-tiny.conv.11](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)
    - for `enet-coco.cfg (EfficientNetB0-Yolov3)` (14 MB): [enetb0-coco.conv.132](https://drive.google.com/file/d/1uhh3D6RSn0ekgmsaTcl-ZW53WBaUDo6j/view?usp=sharing)


### Steps for training and testing the model.

1.  To build the Docker Image `docker-compose -f docker_darknet.yaml build` and than run it `docker-compose -f docker_darknet.yaml up`.

2. Before runing the train and test commands, you must enter inside the Docker Container: `docker exec -it docker_id /bin/bash`.

3. The `docker_id` can be found with: `docker ps`.

4. Once inside the container, you can run the train command: `darknet detector train data/obj.data cfg/yolov3-tiny-prn.cfg files/yolov3-tiny.conv.11 -dont_show -map`

5. (optional) After the training process has finished, you can test its performance with: `darknet detector test data/obj.data cfg/yolov3-tiny-prn.cfg backup/yolov3-tiny-prn_final.weights path_to_the_image -dont-show`

6. (recommended) You can check its performance using my function as well, `yolo_test.py`. Remember to change the line 21: `cap = cv2.VideoCapture('yoko_wakare.mp4')` and change the name of the video for the one you have.

    6.1 Than run `python yolo_test.py` on terminal.
