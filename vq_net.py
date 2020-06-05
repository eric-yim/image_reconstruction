import sonnet as snt
import tensorflow as tf
def residual_block(main_channel, residual_hiddens):
    output = snt.Sequential([
        snt.Conv2D(
            output_channels=residual_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1)),
        tf.nn.relu,
        snt.Conv2D(
            output_channels=main_channel,
            kernel_shape=(1, 1),
            stride=(1, 1)),
        tf.nn.relu,
    ])
    return output
class Encoder(snt.Module):
  def __init__(self, main_channel,num_res_blocks, residual_hiddens,the_stride):
    super(Encoder,self).__init__()

    if the_stride==4:
        self.model = snt.Sequential([
            snt.Conv2D(
                output_channels=main_channel // 2,
                kernel_shape=(4, 4),
                stride=(2, 2),
                name="enc_0"),tf.nn.relu,
            snt.Conv2D(
                output_channels=main_channel,
                kernel_shape=(4, 4),
                stride=(2, 2),
                name="enc_1"),tf.nn.relu,
            snt.Conv2D(
                output_channels=main_channel,
                kernel_shape=(3, 3),
                stride=(1, 1),
                name="enc_2"),tf.nn.relu
        ])
    elif the_stride==2:
        self.model = snt.Sequential([
            snt.Conv2D(
                output_channels=main_channel // 2,
                kernel_shape=(4, 4),
                stride=(2, 2),
                name="enc_0"),tf.nn.relu,
            snt.Conv2D(
                output_channels=main_channel,
                kernel_shape=(3, 3),
                stride=(1, 1),
                name="enc_1"),tf.nn.relu
        ])

    for _ in range(num_res_blocks):
        self.model = snt.Sequential([
              self.model,
              residual_block(main_channel,residual_hiddens)
            ])
  def __call__(self,x):

    h = self.model(x)
    return h
class Decoder(snt.Module):
  def __init__(self, out_channel,dec_channel,num_res_blocks, residual_hiddens,the_stride,
               name='decoder'):
    super(Decoder,self).__init__()
    self.model = snt.Sequential([
        snt.Conv2D(
          output_channels=dec_channel,
          kernel_shape=(3, 3),
          stride=(1, 1),
          name="dec_0"),tf.nn.relu,
    ])
    for _ in range(num_res_blocks):
        self.model = snt.Sequential([
              self.model,
              residual_block(dec_channel,residual_hiddens)
            ])
    if the_stride==4:
        self.model = snt.Sequential([
            self.model,
            snt.Conv2DTranspose(
                output_channels=dec_channel //2,
                output_shape=None,
                kernel_shape=(4, 4),
                stride=(2, 2),
                name="dec_1"),
            tf.nn.relu,
            snt.Conv2DTranspose(
                output_channels=out_channel,
                output_shape=None,
                kernel_shape=(4, 4),
                stride=(2, 2),
                name="dec_2"),
        ])
    elif the_stride==2:
        self.model = snt.Sequential([
            self.model,
            snt.Conv2DTranspose(
                output_channels=out_channel,
                output_shape=None,
                kernel_shape=(4, 4),
                stride=(2, 2),
                name="dec_1"),
        ])
        
    
    
  def __call__(self, x):
    x_recon = self.model(x)

    return x_recon
class VQModel(snt.Module):
  def __init__(
      self,
      in_channel = 3,
      main_channel=128,
      num_res_blocks=2,
      residual_hiddens=32,
      embed_dim = 64,
      n_embed =512,
      decay = 0.99,
      commitment_cost = 0.25,
  ):
    super(VQModel,self).__init__()
    self.enc_b = Encoder(main_channel,num_res_blocks, residual_hiddens,the_stride=4)
    self.enc_t = Encoder(main_channel,num_res_blocks, residual_hiddens,the_stride=2)
    self.quantize_conv_t = snt.Conv2D(
                output_channels=embed_dim,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name="enc_1")
    self.vq_t = snt.nets.VectorQuantizerEMA(
                embedding_dim=embed_dim,
                num_embeddings=n_embed,
                commitment_cost=commitment_cost,
                decay=decay)
    self.dec_t = Decoder(embed_dim,main_channel,num_res_blocks, residual_hiddens,the_stride=2)
    self.quantize_conv_b = snt.Conv2D(
                output_channels=embed_dim,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name="enc_1")
    self.vq_b = snt.nets.VectorQuantizerEMA(
                embedding_dim=embed_dim,
                num_embeddings=n_embed,
                commitment_cost=commitment_cost,
                decay=decay)
    self.upsample_t = snt.Conv2DTranspose(
                output_channels=embed_dim,
                output_shape=None,
                kernel_shape=(4, 4),
                stride=(2, 2),
                name="up_1")
    self.dec = Decoder(in_channel,main_channel,num_res_blocks, residual_hiddens,the_stride=4)
  def __call__(self,x,is_training=False,debug =False):
    tf.cast(x,dtype=tf.float32)
    quant_t,quant_b = self.encode(x,is_training=is_training,debug =debug)
    diff= quant_t["loss"]+quant_b["loss"]
    dec = self.decode(quant_t["quantize"],quant_b["quantize"])
    return dec,diff,quant_t,quant_b
  def encode(self,x,is_training=False,debug =False):
    #Print Shapes for Debugging
    enc_b = self.enc_b(x)
    if debug:
      print("encb", enc_b.shape)
    enc_t = self.enc_t(enc_b)
    if debug:
      print("enct", enc_t.shape)
    quant_t = self.quantize_conv_t(enc_t)
    if debug:
      print("qt", quant_t.shape)
    quant_t = self.vq_t(quant_t,is_training=is_training)
    quant_t_vec = quant_t["quantize"]
    if debug:
      print("qtvec", quant_t_vec.shape)
    dec_t= self.dec_t(quant_t_vec)
    if debug:
      print("dect", dec_t.shape)
    enc_b = tf.concat([enc_b,dec_t],-1)
    quant_b = self.quantize_conv_b(enc_b)
    if debug:
      print("qb", quant_b.shape)
    quant_b = self.vq_b(quant_b,is_training=is_training)
    if debug:
      print("qbvec", quant_b["quantize"].shape)
    return quant_t,quant_b
  def decode(self,quant_t_vec,quant_b_vec):
    upsample_t = self.upsample_t(quant_t_vec)
    quant = tf.concat([upsample_t,quant_b_vec],-1)
    dec = self.dec(quant)
    return dec
  def decode_code(self,code_t,code_b):
    quant_t_vec = self.vq_t.quantize(code_t)
    quant_b_vec = self.vq_b.quantize(code_b)
    dec = self.decode(quant_t_vec,quant_b_vec)
    return dec
