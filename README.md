# CSE-527
Introduction to Computer Vision
Finished in Fall 2023

## HW1

### Problem 1.1: Gaussian Filter (15 points)

#### (a)
- Implement a function that convolves an image and a noisy image array with:
  - A 5x5 Gaussian kernel with sigma=1
  - An 11x11 Gaussian kernel with sigma=2
- Use OpenCV’s `filter2D` routine.
- Generate four output images:
  - The image convolved with the 5x5 kernel
  - The image convolved with the 11x11 kernel
  - The noisy image convolved with the 5x5 kernel
  - The noisy image convolved with the 11x11 kernel

#### (b)
- Create a function that takes an image and its noisy version and returns the Peak Signal-to-Noise Ratio (PSNR) value.

### Problem 1.2: Median Filter (15 points)

#### (a)
- Create a function that generates an image with salt and pepper noise.

#### (b)
- Implement a median filter function.
- Apply the median filter to two noisy images corrupted by salt and pepper noise with noise probabilities of 0.1 and 0.2, respectively.
1. Denoise the images using a median filter.
2. Denoise the images using a Gaussian filter.

### Problem 1.3: Bilateral Filter (20 points)

- Implement bilateral filters consisting of a range kernel and a spatial kernel.
- Both kernels should be Gaussian kernels on different domains.
- Obtain unnormalized versions of the two kernels and jointly normalize the combined kernel for each pixel.

## Problem 2: Ray Casting Using the Blinn-Phong Model (60 points)

### Part 1: Compute the Camera Intrinsic Matrix and Camera Rays (15 points)

- Compute the 3x3 camera intrinsic matrix given the camera image size, sensor size, and focal length.
- Use the inverse of the camera matrix to compute the direction of each camera ray.

### Part 2: Compute Ray-Object Intersections (15 points)

- For each camera ray, compute the intersection of the camera ray with each object in the scene.
- Store the intersection information in an Intersection object.
- Implement sphere ray intersection, plane ray intersection, and ray-scene intersection.

### Part 3: Render a Vanilla Ray-Cast Image (15 points)

- Render an image to check the sanity of the ray-object intersection algorithm.
- Set a camera image pixel to black if the camera ray corresponding to it hits an object in the scene; otherwise, set it to white.

### Part 4: Rendering a Lambertian Ray Casting Image (15 points)

- Estimate simple Lambertian shading for a scene.
- Add color, lights, materials, but ignore shadows, environment, and specular reflections.

## HW2

### Problem 1: Training a Network From Scratch (40 points)

#### Problem 1.1 (25 points)

- Define a simple network architecture and add jittering, normalization, and regularization to improve recognition accuracy.
- Experiment with different forms of jittering: mirroring, random upscaling, random rotation, random cropping, etc.

#### Problem 1.2 (15 points)

- Implement three techniques to improve the accuracy of your model:
  - Increase the training data by randomly rotating the training images.
  - Add batch normalization.
  - Use different activation functions (e.g., sigmoid) and model architecture modifications.

### Problem 2: Fine-Tune a Pre-Trained Deep Network (30 points)

#### Problem 2.1 (10 points)

- **Strategy A**: Fine-tune an existing network.
  - Replace the last layer (or more) of an existing network with random weights.
  - Fine-tune a pre-trained AlexNet for this scene classification task.

#### Problem 2.2 (10 points)

- **Strategy B**: Use activations as features.
  - Use 1000 activations as features, instead of hand-crafted features (e.g., bag-of-features representation).
  - Train a classifier (typically a linear SVM) in this 1000-dimensional feature space.
  - Use the activations of the pre-trained network as features to train a one-vs-all SVM for the scene classification task.

#### Problem 2.3 (10 points)

- Fine-tune a ResNet network and compare the performance to AlexNet.
- Explain why ResNet performs better or worse.
- Use a resnet50 model.

### Problem 3: Transformer (30 points)

- Build a Vision Transformer (ViT).

## HW3

### Part 1: Start YOLOv5 and YOLOv8 (10 points)

- Use pre-trained models from the Git repo/libraries to run detection on a set of images.

### Part 2: Fine-Tune DETR to Detect Object Centroids (60 points)

- Complete the 5 missing code blocks in the detection transformer training pipeline.

### Part 3: Implement a UNet-Based Segmentation Model (30 points)

- Write a slightly modified version of the UNet segmentation model.
- Implement a standard UNet model to perform semantic segmentation.
- The input to the model will be RGB images (3xHxW) and the output will be boolean masks (HxW).

## HW4

### Problem 1: Matching Transformed Image Using SIFT Features (10 points)

#### Step 1

- Detect keypoints from the given image using functions from the SIFT class.
- Plot the image with keypoint scale and orientation overlaid.

#### Step 2

- Rotate the image 45 degrees clockwise using `cv2.warpAffine`.
- Extract SIFT keypoints of this rotated image.
- Plot the rotated image with the same keypoint scale and orientation as in Step 1.

#### Step 3

- Match SIFT keypoints of the original and rotated image using `knnMatch` from `cv2.BFMatcher`.
- Discard bad matches using the scale test proposed by D. Lowe in the SIFT paper.
- Plot the filtered good keypoint matches on the image and display it.
- The image should have two images side by side with matching lines between them.

#### Step 4

- Find the affine transformation from the rotated image to the original image using the RANSAC algorithm.
- Use `cv2.findHomography` with the method set to `cv2.RANSAC` to compute the transformation matrix.
- Use this matrix and `cv2.warpPerspective` to transform the rotated image back.
- Display the restored image.

### Problem 2: Scene Stitching Using SIFT Features (20 points)

- Use SIFT features to match and align different views of the scene.
- Pad the center image with zeros to a larger size using `cv2.copyMakeBorder`.
- Extract the SIFT features for all images and perform the same steps as in Problem 1.
- Find the affine transformation between the two images and align one of the images with the other using `cv2.warpPerspective`.
- Use `cv2.addWeighted` to blend the aligned images and display the stitched result.

#### Step 1 (10 points)

- Compute the transformation from the right image to the center image.
- Warp the right image using the computed transformation.
- Stitch the center and right images using alpha blending.
- Show SIFT feature matches between center and right images.
- Show the stitched result (center and right images).

#### Step 2 (5 points)

- Compute the transformation from the left image to the stitched image from Step 1.
- Warp the left image using the computed transformation.
- Stitch the left image from Step 1 and the resulting image together using alpha blending.
- Show SIFT feature matches between the resulting image from Step 1 and the left image.
- Show the final stitched result (all three images).

#### Laplacian (5 points)

- Instead of using `cv2.addWeighted` for blending, implement a Laplacian pyramid to blend two aligned images.

### Problem 3: Stitching a Set of Images (10 points)

- Stitch a set of images without any given order.
