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
parser.add_argument("--test_dir", default='data/test', type=str, help="train directory")
parser.add_argument("--test_output_dir", default='data/test_output', type=str, help="target directory")
parser.add_argument("--ver_name", type=str,default='None',help="Name of Checkpoint, i.e. ver-13")
parser.add_argument("--plot_type", type=str,default='grid',help='Type of Visual',choices={'grid','single'})
parser.add_argument("--chkpt_dir", default='checkpoints/', type=str, help="checkpoint location")

parser.add_argument("--resize_factor", default='2', type=int, help="scale down factor")
parser.add_argument("--model_params_file", default='model_params.json', type=str, help="model params")
parser.add_argument("--train_params_file", default='train_params.json', type=str, help="training params")
parser.add_argument("-o","--operation",default='view_samples',type=str,choices={'view_samples','reconstruct_test'})
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
def get_images(subset='train'):
    global train_data_iter
    if subset == 'train':
        return next(train_data_iter)
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
#STEP 5 VIEW RECONTRUCTIONS
def step5_view_recon():
    chkpt_dir=args.chkpt_dir
    ver_name = args.ver_name
    plot_type = args.plot_type
    model=step2_load_model()
    if ver_name =='None':
        fpath = tf.train.latest_checkpoint(chkpt_dir)
    else:
        fpath = os.path.join(chkpt_dir,ver_name)
    checkpoint=tf.train.Checkpoint(module=model)
    checkpoint.restore(fpath)

    plot_recon(model,type=plot_type)    
def plot_recon(model,type='grid'):
    global train_data_iter
    #global valid_data_iter
    train_data_iter = step1_load_data()
    train_originals = get_images('train')
    train_images,train_targets = train_originals['images'],train_originals['targets']
    train_recons = do_x_recon(vae_all,train_images)
    #valid_originals = get_images('valid')
    #valid_reconstructions = do_x_recon(vae_all,valid_originals)
    if type=='grid':
        f = plt.figure(figsize=(12,6))
        fig_divs = (1,3)
        plt.axis('off')
        plot_ax(train_images.numpy(),
                    'training images',
                    f,
                    1,
                    fig_divs,
                    type=type)
        plot_ax(train_targets.numpy(),
                    'labeled targets',
                    f,
                    2,
                    fig_divs,
                    type=type)
        plot_ax(train_recons.numpy(),
                    'training data reconstructions',
                    f,
                    3,
                    fig_divs,
                    type=type)
        plt.show()
    elif type=='single':
        for a,b,c in zip(train_images.numpy(),train_targets.numpy(),train_recons.numpy()):
            f = plt.figure(figsize=(12,6))
            fig_divs = (1,3)
            plt.axis('off')
            plot_ax(a,
                        'training images',
                        f,
                        1,
                        fig_divs,
                        type=type)
            plot_ax(b,
                        'labeled targets',
                        f,
                        2,
                        fig_divs,
                        type=type)
            plot_ax(c,
                        'training data reconstructions',
                        f,
                        3,
                        fig_divs,
                        type=type)
            plt.show()
    elif type=='present':
        for a,b,c in zip(train_images.numpy(),train_targets.numpy(),train_recons.numpy()):
            f = plt.figure(figsize=(12,4))
            fig_divs = (1,2)
            plt.axis('off')
            plot_ax(a,
                        'Training Image',
                        f,
                        1,
                        fig_divs,
                        type=type)

            plot_ax(c,
                        'Crowd Estimation',
                        f,
                        2,
                        fig_divs,
                        type=type)
            plt.show()
        
    


def do_x_recon(model,x):
    x_recon,_,_,_=model(x,is_training=False)
    return x_recon
def convert_batch_to_image_grid(image_batch):
    n_v =2
    n_h = 2
    w,h = image_batch.shape[1:3]
    larger = image_batch[0:4]
    reshaped = (larger.reshape(n_v, n_h, w, h, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(n_h * w, n_v * h, 3))
    return reshaped + 0.5
def plot_ax(x,the_str,f,i,fig_divs=(1,2),type='grid'):
    ax=f.add_subplot(*fig_divs,i)
    if type=='grid':
        ax.imshow(convert_batch_to_image_grid(x),
                  interpolation='nearest')
    elif type=='single' or type=='present':
        ax.imshow(x+ 0.5,interpolation='nearest')
    ax.set_title(the_str)
    
    
def plot_recon(model,type='grid'):
    global train_data_iter
    #global valid_data_iter
    train_data_iter = step1_load_data()
    train_originals = get_images('train')
    train_images,train_targets = train_originals['images'],train_originals['targets']
    train_recons = do_x_recon(model,train_images)
    #valid_originals = get_images('valid')
    #valid_reconstructions = do_x_recon(vae_all,valid_originals)
    if type=='grid':
        f = plt.figure(figsize=(12,6))
        fig_divs = (1,3)
        plt.axis('off')
        plot_ax(train_images.numpy(),
                    'training images',
                    f,
                    1,
                    fig_divs,
                    type=type)
        plot_ax(train_targets.numpy(),
                    'labeled targets',
                    f,
                    2,
                    fig_divs,
                    type=type)
        plot_ax(train_recons.numpy(),
                    'training data reconstructions',
                    f,
                    3,
                    fig_divs,
                    type=type)
        plt.show()
    elif type=='single':
        for a,b,c in zip(train_images.numpy(),train_targets.numpy(),train_recons.numpy()):
            f = plt.figure(figsize=(12,6))
            fig_divs = (1,3)
            plt.axis('off')
            plot_ax(a,
                        'training images',
                        f,
                        1,
                        fig_divs,
                        type=type)
            plot_ax(b,
                        'labeled targets',
                        f,
                        2,
                        fig_divs,
                        type=type)
            plot_ax(c,
                        'training data reconstructions',
                        f,
                        3,
                        fig_divs,
                        type=type)
            plt.show()
    elif type=='present':
        for a,b,c in zip(train_images.numpy(),train_targets.numpy(),train_recons.numpy()):
            f = plt.figure(figsize=(12,4))
            fig_divs = (1,2)
            plt.axis('off')
            plot_ax(a,
                        'Training Image',
                        f,
                        1,
                        fig_divs,
                        type=type)

            plot_ax(c,
                        'Crowd Estimation',
                        f,
                        2,
                        fig_divs,
                        type=type)
            plt.show()
        
    

#===============================================================
#STEP 6 TEST RECONSTRUCTION
def get_img_names_6():
    test_dir = args.test_dir
    listing = os.listdir(test_dir)
    listing_orig = [os.path.join(test_dir,item) for item in listing]
    return listing_orig,listing
def load_data_6():
    list_orig,_=get_img_names_6()
    train_orig = list_to_batch(list_orig)
    return train_orig
def write_data_6(x_recon):
    test_output_dir = args.test_output_dir
    _,listing=get_img_names_6()
    for x,li in zip(x_recon,listing):
        #x= x+0.5
        #print(np.min(x),np.max(x))
        #plt.imshow(x+0.5)
        #print(x+0.5)
        
        #plt.show()
        x = reverse_norm(x)
        temp_dir = os.path.join(test_output_dir,li)
        im = Image.fromarray(x)
        im.save(temp_dir)

def step6_test_recon():
    chkpt_dir = args.chkpt_dir
    ver_name = args.ver_name
    global train_params
    batch_size = train_params['batch_size']
    model=step2_load_model()
    if args.ver_name =='None':
        fpath = tf.train.latest_checkpoint(chkpt_dir)
    else:
        fpath = os.path.join(chkpt_dir,ver_name)
    checkpoint=tf.train.Checkpoint(module=model)
    checkpoint.restore(fpath)

    x = load_data_6()
    #Needs to go in batches to prevent OOM
    ds_iterator = iter(
        tf.data.Dataset.from_tensor_slices(x)
        .map(cast_norm)
        .repeat(1)  # repeat indefinitely
        .batch(batch_size))  
    num_iter = len(x) // batch_size +(0 if (len(x)%batch_size)==0 else 1)
    results = []
    for _ in range(num_iter):
        
        x0 = next(ds_iterator)
        x_recon = do_x_recon(model,x0)
        results.append(x_recon.numpy())
    results = np.concatenate(results,axis=0)
    write_data_6(results)


#===============================================================
#MAIN
def main():
    
    if args.operation=='view_samples':
        global train_data_iter
        train_data_iter = step1_load_data()
        step5_view_recon()
    elif args.operation=='reconstruct_test':
        step6_test_recon()
if __name__=="__main__":
    main()
