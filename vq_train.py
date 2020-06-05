import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from vq_net import VQModel
import json
import argparse
#
"""
Purpose: reconstruct images to a target image.
This can be used for VAEs, image restorations, or image segmentations.

Usage: python vq_train.py
"""
#ArgParse#TODO
parser = argparse.ArgumentParser()

parser.add_argument("--train_dir", default='data/originals', type=str, help="train directory")
parser.add_argument("--target_dir", default='data/targets', type=str, help="target directory")
parser.add_argument("--chkpt_dir", default='checkpoints/', type=str, help="checkpoint location")

parser.add_argument("--resize_factor", default='2', type=int, help="scale down factor")
parser.add_argument("--model_params_file", default='model_params.json', type=str, help="model params")
parser.add_argument("--train_params_file", default='train_params.json', type=str, help="training params")
parser.add_argument("--debug", dest='debug', action='store_true', help="print debug statements")

args = parser.parse_args()

with open(args.train_params_file,'r') as f:
    train_params = json.loads(f.read())
with open(args.model_params_file,'r') as f:
    model_params = json.loads(f.read())


#===============================================================
#STEP1 LOAD DATA
def get_img_names():
    """
    Get a list of images in train and target directories
    """
    train_dir = args.train_dir
    target_dir = args.target_dir
    listing = os.listdir(train_dir)
    listing_orig = [os.path.join(train_dir,item) for item in listing]
    listing_target = [os.path.join(target_dir,item) for item in listing]
    return listing_orig,listing_target
def list_to_batch(img_list):
    """
    Converts list of images to np array
    """
    resize_factor = args.resize_factor
    imgs = [Image.open(imgname).convert('RGB') for imgname in img_list]
    w,h = imgs[0].size
    imgs = [np.expand_dims(np.array(img.resize((w//resize_factor,h//resize_factor))),axis=0) for img in imgs]
    imgs = np.concatenate(imgs,axis=0)
    return imgs
#===============================================================
#FORMAT DATA
def cast_norm(images):
    return (tf.cast(images, tf.float32) / 255.0) - 0.5
def reverse_norm(images):
    images = np.clip(np.round((images + 0.5) * 255.0),0,255).astype(np.uint8)
    return images
    
def cast_and_normalise_images(data_dict):
    """
    Convert images to floating point with the range [-0.5, 0.5]
    Convert targets to floating point with the range [-0.5, 0.5]
    """
    images = data_dict['images']
    targets = data_dict['targets']
    data_dict['images'] = cast_norm(images)
    data_dict['targets'] = cast_norm(targets) 
    return data_dict  
def step1_load_data():
    """
    Load data into tf dataset iterator
    """
    global train_params
    global data_variance
    batch_size = train_params['batch_size']
    list_orig,list_targ=get_img_names()
    train_orig,train_targ = list_to_batch(list_orig),list_to_batch(list_targ)
    train_data_dict = {'images':train_orig,'targets':train_targ}
    data_variance = np.var(train_orig / 255.0)
    train_dataset_iterator = iter(
        tf.data.Dataset.from_tensor_slices(train_data_dict)
        .map(cast_and_normalise_images)
        .shuffle(10000)
        .repeat(-1)  # repeat indefinitely
        .batch(batch_size))

    return train_dataset_iterator
#===============================================================
#STEP 2 LOAD MODEL
def step2_load_model():
    global model_params
    model = VQModel(in_channel = 3,
        **model_params)
    return model
#===============================================================
#STEP 3 TRAIN
def get_images(subset='train'):
    global train_data_iter
    if subset == 'train':
        return next(train_data_iter)


@tf.function
def train_step(model,opt,x,x_targ):
    global data_variance
    with tf.GradientTape() as tape:

        x_recon,diff,quant_t,quant_b=model(x,is_training=True)
        recon_error = tf.reduce_mean((x_recon - x_targ)**2) / data_variance
        loss = diff+recon_error
    params= model.trainable_variables
    grads = tape.gradient(loss, params)
    opt.apply_gradients(zip(grads, params))
    return recon_error,quant_t["perplexity"],quant_b["perplexity"]
def step3_train(model):
    global train_params
    results = train(model,**train_params)
    return results

def train(model,
        num_training_updates,
        batch_size=32,
        learning_rate=1e-4,
        ):
    chkpt_dir = args.chkpt_dir
    checkpoint = tf.train.Checkpoint(module = model)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_res_recon_error = []
    train_res_perplexity_t = []
    train_res_perplexity_b = []
    for i in range(num_training_updates):
        inx = get_images('train')
        x,x_targ = inx['images'],inx['targets']
        results = train_step(model,opt,x,x_targ)
        train_res_recon_error.append(results[0])
        train_res_perplexity_t.append(results[1])
        train_res_perplexity_b.append(results[2])
        if (i+1) % 100 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity_t[-100:]))
            print('perplexity: %.3f\n' % np.mean(train_res_perplexity_b[-100:]))
            checkpoint.save(os.path.join(chkpt_dir,'ver'))
    if not (i+1) % 100 == 0:
        checkpoint.save(os.path.join(chkpt_dir,'ver'))
    return train_res_recon_error,train_res_perplexity_t,train_res_perplexity_b
#===============================================================
#STEP 4 PLOT TRAIN
def step4_plot_train(results):
    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,3,1)
    ax.plot(results[0])
    ax.set_yscale('log')
    ax.set_title('NMSE.')

    ax = f.add_subplot(1,3,2)
    ax.plot(results[1])
    ax.set_title('Top: Avg codebook usage (perplexity).')
    ax = f.add_subplot(1,3,3)
    ax.plot(results[2])
    ax.set_title('Bottom: Avg codebook usage (perplexity).')
    plt.show()



#===============================================================
#MAIN
def main():
    #Load Data into Iterator
    global train_data_iter
    train_data_iter = step1_load_data()
    #Load Net
    model = step2_load_model()
    #Train
    results=step3_train(model)
    #Plot Train
    step4_plot_train(results)
    
        

main()
