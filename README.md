# WDICD

Open access of simulation satellite segmentation and pose estimation dataset: WDICD

step1: download the 3d satellite model in NASA and install blender

step2: change the model path in the code

step3: create the environment follow pcn

step4: Run `blender -b -P render_rgbd_imagefuse.py [ShapeNet directory] [model list] [output directory] [num scans per model]` to render the depth images. The images will be stored in OpenEXR format.

step5. Run `python3 process_exr.py [model list] [intrinsics file] [output directory] [num scans per model]` to convert the `.exr` files into 16 bit PNG depth images and point clouds in the model's coordinate frame.
