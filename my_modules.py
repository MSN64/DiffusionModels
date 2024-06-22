import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 32

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema = 2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



class SelfAttention(nn.Module):                 #normal attention block, this is different from skip connections
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first = True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(           #add skip-connection and pass it through feedforward layer, also consisting of a layer norm
            nn.LayerNorm([channels]),           #and two linear layers which are separated by a GELU activation (weights the input by its probability under a Gaussian distribution)
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        #print(x.shape)
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)              #first and last operations for bringing image into the right shape
        #by first flattening them, and then bringing the channel axis as the last dimension such that the attention can work properly
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):                                                                   #normal conv. block
    def __init__(self, in_channels, out_channels, mid_channels = None, residual = False):      #possibility to add a residual connection with 'True'
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1, bias = False),  #2d convolution
            nn.GroupNorm(1, mid_channels),                                           #followed by a group norm
            nn.GELU(),                                                                         #and a GELU activation
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1, bias = False), #another 2d convolution and group norm
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):                                        #downsample block
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),                                  #maxpool component (2x2) to reduce the size by half, followed by two double convolutions
            DoubleConv(in_channels, in_channels, residual = True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),                                        #consists of a SiLU activation
            nn.Linear(                                        #linear layer going from time embedding to the hidden dimension
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)                                                          #first feed images through the convolutional block
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  #project the time embedding accordingly to proper dim.
        #print("Down x shape: ", x.shape)
        #print("Down emb shape: ", emb.shape)
        return x + emb                                                                    #then add both together and return


#upsample block, also take in the skip-connection which comes from the encoder
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()

        self.up = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)
        self.conv = nn.Sequential(
            #after upsampling the normal x, we concatenate it with the skip-connection and feed it through the convolutional block
            DoubleConv(in_channels, in_channels, residual = True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        #at the end, also add the time embedding layer to it again
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):                                  #take in skip connection from the encoder skip_x
        x = self.up(x)                                                #upsample image
        x = torch.cat([skip_x, x], dim = 1)                    #concatenate x with skip connection
        x = self.conv(x)                                              #feed it to convolutional block
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        #print("Up x shape: ", x.shape)
        #print("Up emb shape: ", emb.shape)
        return x + emb




class UNet(nn.Module):
    def __init__(self, c_in = 1, c_out = 1, time_dim = 256, remove_deep_conv=False, device = "cuda", img_size = IMAGE_SIZE):
        super().__init__()                                 #unit has encoder, bottleneck and decoder
        self.device = device
        self.time_dim = time_dim
        self.img_size = img_size
        self.remove_deep_conv = remove_deep_conv

        #encoder
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)

        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)

        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)


        #bottleneck, consisting of a bunch of convolutional layers
        if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)
        #print("Channels:", self.bot3)


        #decoder
        self.up1 = Up(512, 128)                   #three upsample blocks, each followed by a self-attention block
        #print("Channels", self.up1)
        self.sa4 = SelfAttention(128)

        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)

        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        #Change output number to 1 for MSE loss case
        self.outc = nn.Conv2d(64, c_out * 2, kernel_size = 1)             #projecting back to the output channel dimension using a normal conv. layer, 64 is just channel numbers, which converts to 1 as output. Changed to 2 due to mean and var outputs


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device = one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim = -1)
        return pos_enc


    def unet_forward(self, x, t):                               #take as input the noise images, and timesteps t (tensor with integer timestep values in it)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        #print("After Down1 shape: ", x2.shape)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        #print("After Down2 shape: ", x3.shape)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        #print("After Down3 shape: ", x4.shape)
        x4 = self.sa3(x4)
        #print("Bottleneck input shape: ", x4.shape)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        #print("Bottleneck x4: ", x4.shape)

        x = self.up1(x4, x3, t)                           #upsampling blocks take in skip connections from the encoder (x3)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        #remove the next few lines and just return output for use of MSE loss case
        noise_pred, log_var = torch.chunk(output, 2, dim = 1)   #splitting output into noise and log variance

        # Adjust log_var to have the size (1, 1, 1, 1)
        log_var = torch.mean(log_var, dim=[2, 3], keepdim=True)                                                         #is this a good way to resize the tensor to this size?
        return noise_pred, log_var

    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)

model_params = UNet()
total_params = sum(p.numel() for p in model_params.parameters())
print(f"Number of parameters: {total_params}")


if __name__ == '__main__':
    net = UNet(device = "cpu")                                                                                          #change to CUDA?
    print("Total number of parameters:", sum([p.numel() for p in net.parameters()]))
    x = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    t = x.new_tensor([500] * x.shape[0]).long()
    print(net(x, t).shape)