# Tensorflow 2.0 Implementation of Image Reconstruction Network

See https://arxiv.org/abs/1906.00446

<img src='https://user-images.githubusercontent.com/48815706/83919525-9dbe0900-a72f-11ea-8c71-0c6ad014cdf9.png'>

<p>In the paper, the authors concatenate layers at multiple resolutions, similar to a U-net. The main difference is an explicit latent space. Here, they use the network as a VAE, but generating samples from in the latent space and upsampling.
  </p>
<p>The network itself can be repurposed for many image processing tasks: for example image restoration or image segmentation - or any task that starts with an input image and outputs some target image.</p>
<h2>Usage</h2>
<h5>File Structure</h5>
<p>Place original images in path data/originals</p>
<p>Place target images in path data/targets</p>
<p>Run</p>
```
python vq_train.py
```
