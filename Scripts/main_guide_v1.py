import sys
import time
import math 
import argparse
import numpy as np
from imageio import imsave
from pathlib import Path
from matplotlib import pyplot as plt
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Data import TrainData, TestData, TestData_quant
from Models import Network, LoopFilter, LoopFilter_UV
from Models import Network_guide_v1
from Utils import Checkpoints, LineFit


# Command parameters
parser = argparse.ArgumentParser()
parser.add_argument("mode", type=str, help="Mode in [train|test]")
parser.add_argument("--not_resume", action="store_true", default=False, help="Resume the checkpoint")
parser.add_argument("--profile", type=str, default="very high", help="Coding profile")
parser.add_argument("--batchsize", type=int, default=4, help="Batch size")
parser.add_argument("--patchsize", type=int, default=256, help="Patch size")
parser.add_argument("--last_steps", type=int, default=5 * 10 ** 6, help="Train up to this number of steps.")
opts = parser.parse_args()

# 原本模型是very high_guide_v1 现在用的是云脑的版本 very high_guide_v1_server用于后面的训练
# Global variables
g_types = {"dtype": torch.float32, "device": torch.device("cuda:0")}
g_profiles = {
    "super high": (0.06, 192),
    "very high": (0.045, 192),
    "high": (0.03, 192),
    "a little high": (0.015, 192),
}


class BasicOps(object):
    def __init__(self, options):
        super().__init__()
        # Define the hyper-parameters
        self.factor = g_profiles[options.profile][0]
        self.channels = g_profiles[options.profile][1]
        self.log = Path(__file__).resolve().parent / options.profile
        self.checkpoints = Checkpoints(10, self.log)
        self.patch_size = options.patchsize

        # Define the model
        self.network = Network_guide_v1(self.channels, context=True).to(**g_types)
        self.loop_filter = LoopFilter(64).to(**g_types)
        self.loop_filter_UV = LoopFilter_UV(64).to(**g_types)
        # Print the summary
        self._print_summary()

    def _print_summary(self):
        print("======= SUMMARY =======")
        print("Parameters: " + format(sum(item.numel() for item in self.network.parameters()), ','))
        print("=======================\n")

    def _load(self):
        
        checkpoint = torch.load(self.checkpoints.file)
        self.network.load_state_dict(checkpoint["network"])
        checkpoint_new2 = torch.load("very high_guide_v1_server/checkpoint-8064000.pth")
        self.loop_filter.load_state_dict(checkpoint_new2["loop_filter_Y"])
        checkpoint_new = torch.load("high_guide_v1_server/checkpoint-10598000.pth")
        self.loop_filter_UV.load_state_dict(checkpoint_new["loop_filter_UV"])
        
        return checkpoint["factor"]
    
    @staticmethod
    def unsqueeze(x, r):
        return F.pixel_shuffle(x, r)

    @staticmethod
    def squeeze(x, r):
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H // r, r, W // r, r)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * (r ** 2), H // r, W // r)

        return x

    def forward_one_pass(self, y_comp, u_comp, v_comp):

        time1 = time.time()
        results = self.network(y_comp, u_comp, v_comp)
        time2 = time.time()
        cost_time = time2-time1
        outputs_Y = results["Y"]
        outputs_UV = results["UV"]
        outputs_U = outputs_UV[:, 0:1, :, :]
        outputs_V = outputs_UV[:, 1:2, :, :]
        bpp = results["bpp"]

        outputs_Y = self.loop_filter(outputs_Y)
        outputs_Y_ = self.squeeze(outputs_Y, 2)
        outputs_filter = torch.cat([outputs_Y_, outputs_UV], 1)
        outputs_filter = self.loop_filter_UV(outputs_filter)
               
        outputs_U = outputs_filter[:,4:5,:,:]
        outputs_V = outputs_filter[:,5:6,:,:]
        # outputs_U = outputs_UV[:, 0:1, :, :]
        # outputs_V = outputs_UV[:, 1:2, :, :]

        # outputs_U = outputs_U.unsqueeze(dim=1)
        # outputs_Y = outputs_filter[:,0:4,:,:]
        # outputs_Y = self.unsqueeze(outputs_Y, 2)

        mse_y = torch.mean(((outputs_Y - y_comp) * 255.0) ** 2)
        mse_u = torch.mean(((outputs_U - u_comp) * 255.0) ** 2)
        mse_v = torch.mean(((outputs_V - v_comp) * 255.0) ** 2)

        loss = self.factor * (6/8*mse_y + 1/8*mse_u + 1/8*mse_v) + bpp

        return loss, mse_y, bpp, (outputs_Y, outputs_U, outputs_V), mse_u, mse_v, cost_time


class TrainingProc(BasicOps):
    def __init__(self, options):
        super().__init__(options)

        # Open the CuDNN_FIND
        # duan modify remove the acceleration module
        # torch.backends.cudnn.benchmark = True

        # Define the training parameters
        self.not_resume = options.not_resume
        self.batch_size = options.batchsize
        self.last_steps = options.last_steps
        self.writer = SummaryWriter(self.log)
        self.options = options

        # Define the important tools
        self.train_loader = None
        self._init_loaders()
        self._init_models()

        self.main_optimizer = None
        self._init_optimizers()

    def _init_loaders(self):
        self.train_loader = DataLoader(
            dataset=TrainData(self.patch_size, debug=False),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def _init_models(self):

        self.test_case = TestingProc(self.network, self.loop_filter, self.loop_filter_UV, False, self.options)

    def _init_optimizers(self):
        network_params = list()
        integer_offset = list()
        lf_params = list()
        # lf_params = self.loop_filter.parameters()
        network_params = self.network.parameters()

        # lf_para = [item for item in self.loop_filter.parameters()]
        network_para = [item for item in self.network.parameters()]
        optim_params = list()
        # optim_params = optim_params + lf_para + network_para
        # last phase
        optim_list = list()
        decoder_para = [item for item in self.network.decoder.parameters()]
        hyper_decoder_para = [item for item in self.network.hyper_decoder.parameters()]
        context_model_para = [item for item in self.network.context_model.parameters()]
        entropy_para = [item for item in self.network.entropy_parameters.parameters()]
        factor_para = [item for item in self.network.factorized.parameters()]
        condition_para = [item for item in self.network.GMM.parameters()]

        lf_params = list()
        lf_Y = [item for item in self.loop_filter.parameters()]
        lf_UV = [item for item in self.loop_filter_UV.parameters()]
        # lf_U = [item for item in self.loop_filter_U.parameters()]
        # lf_V = [item for item in self.loop_filter_V.parameters()]
        lf_params += lf_Y
        # lf_params += lf_UV
        self.main_optimizer = optim.Adam(network_params, lr=3*(1e-5))
        # self.main_optimizer = optim.Adam(optim_list, lr=3*(1e-5))
        # self.main_optimizer = optim.Adam(lf_params, lr=5*(1e-5))
        # self.main_optimizer = optim.Adam(optim_params, lr=2*(1e-5))


    def _load(self):
        checkpoint = torch.load(self.checkpoints.file)
        self.network.load_state_dict(checkpoint["network"])
        self.loop_filter.load_state_dict(checkpoint["loop_filter_Y"])
        self.loop_filter_UV.load_state_dict(checkpoint["loop_filter_UV"])

        return checkpoint["iteration"]

    def _save(self, iteration):
        ckpt_file = self.log / "checkpoint-{}.pth".format(iteration)
        checkpoint = {
            "factor": self.factor,
            "iteration": iteration,
            "network": self.network.state_dict(),
            'loop_filter_Y': self.loop_filter.state_dict(),
            # 'loop_filter_U': self.loop_filter_U.state_dict(),
            'loop_filter_UV': self.loop_filter_UV.state_dict(),
            "main_optimizer": self.main_optimizer.state_dict(),
        }
        self.checkpoints.save(ckpt_file, checkpoint)

    def _get_images(self):
        while True:
            for images in self.train_loader:
                yield images
    @staticmethod
    def set_trainable(module, requires_grad):
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def find_lr(self, init_lr=1e-8, final_lr=1e2, beta=0.5):
        # Set the initial learning rate
        factor = (final_lr / init_lr) ** (1 / 500)
        scheduler = optim.lr_scheduler.ExponentialLR(self.main_optimizer, factor)
        for param_group in self.main_optimizer.param_groups:
            param_group["lr"] = init_lr

        # Define the records
        avg_loss = 0.0
        best_loss = 0.0
        batch_num = 0
        losses = []
        lrs = []

        self.network.train()
        for index, inputs in enumerate(self._get_images()):
            batch_num += 1

            y_comp, u_comp, v_comp = [item.to(**g_types, non_blocking=True) for item in inputs]
            
            loss, _, _, _ = self.forward_one_pass(y_comp, u_comp, v_comp)

            lr = max(param_group["lr"] for param_group in self.main_optimizer.param_groups)
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break
            if lr > final_lr:
                break
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            lrs.append(lr)
            losses.append(smoothed_loss)

            self.main_optimizer.zero_grad()
            loss.backward()
            self.main_optimizer.step()
            scheduler.step()

        plt.figure()
        plt.semilogx(lrs, losses)
        plt.grid(True)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.show()

    def run(self):
        # Load the checkpoint
        if self.not_resume or not self.checkpoints.is_exists():
            iteration = 0
        else:
            iteration = self._load()

        # fix the compression/lf model 1-3 phase modify in 3.2
        # self.set_trainable(self.network, True)
        # self.set_trainable(self.loop_filter, True)
        
        # self.set_trainable(self.loop_filter_UV, True)
        # self.set_trainable(self.loop_filter, True)
        # # as follows are train for 4 phase
        self.set_trainable(self.network.encoder, False)
        self.set_trainable(self.network.decoder, True)
        self.set_trainable(self.network.hyper_encoder, False)
        self.set_trainable(self.network.hyper_decoder, True)
        self.set_trainable(self.network.context_model, True)
        self.set_trainable(self.network.entropy_parameters, True)
        self.set_trainable(self.network.factorized, True)
        self.set_trainable(self.network.GMM, True)
        # self.set_trainable(self.loop_filter, True)

        
        file_print = open("test.txt", "w")
        # Enter the loop
        pts = time.perf_counter()  # Presentation Time Stamp
        loss_min = 0.843039
        # for name, param in self.network.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # for name, param in self.loop_filter_U.named_parameters():
        #     if param.requires_grad:
        #         print("yes")
        #         print(name)      
        for inputs in self._get_images():
            iteration += 1

            # Set the network mode
            self.network.train()
            # self.loop_filter.train()
            # self.loop_filter_U.train()
            self.loop_filter.train()
            self.loop_filter_UV.train()
            if hasattr(torch.cuda, 'empty_cache'):
    	        torch.cuda.empty_cache()

            # Run the forward pass
            y_comp, u_comp, v_comp = [item.to(**g_types, non_blocking=True) for item in inputs]
            loss, mse, rate, _, mse_u, mse_v = self.forward_one_pass(y_comp, u_comp, v_comp)

            # Run the backward pass
            self.main_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.network.parameters(), max_norm=10)
            self.main_optimizer.step()

            # Check whether is inf/NaN
            if torch.isinf(loss.detach()) or torch.isnan(loss.detach()):
                raise ValueError("Loss has been Inf/NaN. (Iter {})".format(iteration))

            # Test the model and write the log
            if iteration % 200 == 0:
                # Compute the training loss

                results = self.test_case.run()
                psnr_y_test  = results["psnr_y"]
                psnr_u_test  = results["psnr_u"]
                psnr_v_test  = results["psnr_v"]
                loss_test = results["loss"]
                rate_test = results["bpp"]

                train_lr = max(param_group["lr"] for param_group in self.main_optimizer.param_groups)
                train_loss = loss.item()
                train_mse = mse.item()
                train_mse_U = mse_u.item()
                train_mse_V = mse_v.item()
                train_rate = rate.item()
                train_psnr = 10 * math.log10(255 ** 2 / train_mse)
                train_psnr_U = 10 * math.log10(255 ** 2 / train_mse_U)
                train_psnr_V = 10 * math.log10(255 ** 2 / train_mse_V)

                # Print the results
                print("[iter {:<5d}] ".format(iteration), end='\t')
                print("LR: {:>3.1e} ".format(train_lr), end='\t')
                print("Loss: {:>.4f} ".format(train_loss), end='\t')
                print("PSNR_Y: {:>05.2f} dB ".format(train_psnr), end='\t')
                print("PSNR_U: {:>05.2f} dB ".format(train_psnr_U), end='\t')
                print("PSNR_V: {:>05.2f} dB ".format(train_psnr_V), end='\t')
                print("Rate: {:>06.4f} bpp ".format(train_rate), end='\t')
                print("Time: {:>.2f} s ".format(time.perf_counter() - pts))
                print("!!! Test_PSNR_Y: {:>05.2f} dB ".format(psnr_y_test), end='\t')
                print("!!! Test_PSNR_U: {:>05.2f} dB ".format(psnr_u_test), end='\t')
                print("!!! Test_PSNR_V: {:>05.2f} dB ".format(psnr_v_test), end='\t')
                print("!!! Test loss: {:>.10f} ".format(loss_test), end='\t')
                print("!!! Test_Rate: {:>06.4f} bpp ".format(rate_test))
                pts = time.perf_counter()

                # self.writer.add_scalar("LR", train_lr, global_step=iteration)
                print("!!! Test_PSNR_Y: {:>05.2f} dB ".format(psnr_y_test), end='\t', file = file_print)
                print("!!! Test_PSNR_U: {:>05.2f} dB ".format(psnr_u_test), end='\t', file = file_print)
                print("!!! Test_PSNR_V: {:>05.2f} dB ".format(psnr_v_test), end='\t', file = file_print)
                print("!!! Test_Rate: {:>06.4f} bpp ".format(rate_test), file = file_print)
                self.writer.add_scalar("Training Loss", train_loss, global_step=iteration)
                self.writer.add_scalar("Training PSNR_Y", train_psnr, global_step=iteration)
                self.writer.add_scalar("Training PSNR_U", train_psnr_U, global_step=iteration)
                self.writer.add_scalar("Training PSNR_V", train_psnr_V, global_step=iteration)
                self.writer.add_scalar("Training Rate", train_rate, global_step=iteration)
                self.writer.add_scalar("Y MSE", train_mse, global_step=iteration)
                self.writer.add_scalar("U MSE", train_mse_U, global_step=iteration)
                self.writer.add_scalar("V MSE", train_mse_V, global_step=iteration)

                self.writer.add_scalar("Y test_psnr", psnr_y_test, global_step=iteration)
                self.writer.add_scalar("U test_psnr", psnr_u_test, global_step=iteration)
                self.writer.add_scalar("V test_psnr", psnr_v_test, global_step=iteration)
                self.writer.add_scalar("Test loss", loss_test, global_step=iteration)
                self.writer.add_scalar("Test Rate", rate_test, global_step=iteration)



            # Save the model
            if iteration % 200 == 0:
                # self._save(iteration)
                if loss_test < loss_min:
                    loss_min = loss_test
                    self._save(iteration)
                    print("update new checkpoint, new_test_loss is: ", loss_test, end='\n')
                    print("update new checkpoint, new_test_loss is: ", loss_test, file = file_print)

            # Finish the training process
            if iteration > self.last_steps:
                self._save(iteration)
                self.writer.flush()
                self.writer.close()
                break

def yuv_export(filename, Y, U, V, skip=1):
    yfrm = Y.shape[0]
    ufrm = U.shape[0]
    vfrm = V.shape[0]

    if yfrm == ufrm == vfrm:
        numfrm = yfrm
        
    
    else:
        raise Exception("The length of the frames does not match.")

    with open(filename, "wb") as f:
        for i in range(numfrm):
            if i % skip == 0:
                f.write(Y[i, :, :].tobytes())
                f.write(U[i, :, :].tobytes())
                f.write(V[i, :, :].tobytes())

def yuv_import(filename, width, height, numfrm, startfrm=0):
    # Open the file
    f = open(filename, "rb")

    # Skip some frames
    luma_size = height * width
    chroma_size = luma_size // 4
    frame_size = luma_size * 3 // 2
    f.seek(frame_size * startfrm, 0)

    # Define the YUV buffer
    Y = np.zeros([numfrm, height, width], dtype=np.uint8)
    U = np.zeros([numfrm, height//2, width//2], dtype=np.uint8)
    V = np.zeros([numfrm, height//2, width//2], dtype=np.uint8)

    # Loop over the frames
    for i in range(numfrm):
        # Read the Y component
        Y[i, :, :] = np.fromfile(f, dtype=np.uint8, count=luma_size).reshape([height, width])
        # Read the U component
        U[i, :, :] = np.fromfile(f, dtype=np.uint8, count=chroma_size).reshape([height//2, width//2])
        # Read the V component
        V[i, :, :] = np.fromfile(f, dtype=np.uint8, count=chroma_size).reshape([height//2, width//2])

    # Close the file
    f.close()

    return Y, U, V

class TestingProc(BasicOps):
    def __init__(self, network, loop_filter, loop_filter_UV,  decoded, options):
        super().__init__(options)
        # Set the configurations
        self.decoded = decoded

        # Define the important tools
        self.test_loader = None
        self._init_dataset()

        self.network = network
        self.loop_filter = loop_filter
        self.loop_filter_UV = loop_filter_UV
        # self.loop_filter_V = loop_filter_V

        self.anchor = None
        self._init_tools()

        # Load the data
        # duan modify 2021.2.12 这个地方这个时候不需要load 因为self.factor和self.network都是赋值进去的 所以不需要
        # 原先那种方式需要是因为这个地方是test阶段 要看才需要
        # self.factor = self._load()

        # Create the output directory
        if self.decoded:
            self.outputs_dir = self.log / "reconstruction"
            if not self.outputs_dir.exists():
                self.outputs_dir.mkdir()
            else:
                print("Directory has existed!")

    def _init_dataset(self):
        test_dataset = TestData()
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )
    
    def _init_tools(self):
        self.anchor = LineFit()

    @staticmethod
    def _norm(x):
        return (x * 255.0).clamp(0, 255).round_()

    def run(self):
        # Begin the training process
        psnr_y_list = list()
        psnr_u_list = list()
        psnr_v_list = list()
        test_loss = list()
        rate_list = list()
        self.network.eval()
        self.loop_filter.eval()
        # self.loop_filter_U.eval()
        # self.loop_filter_V.eval()
        self.loop_filter_UV.eval()
        # self.loop_filter.eval()
        with torch.no_grad():
            for index, inputs in enumerate(self.test_loader):
                # Get the input blocks
                if hasattr(torch.cuda, 'empty_cache'):
    	            torch.cuda.empty_cache()

                y_comp, u_comp, v_comp = [item.to(**g_types, non_blocking=True) for item in inputs]
                
                loss, mse, pixel_bit, outputs, mse_u, mse_v = self.forward_one_pass(y_comp, u_comp, v_comp)
                # print(mse, mse_u, mse_v)
                psnr_y = 10 * torch.log10(255 ** 2 / mse)
                psnr_u = 10 * torch.log10(255 ** 2 / mse_u)
                psnr_v = 10 * torch.log10(255 ** 2 / mse_v)
                # print(psnr)
                psnr_y_list.append(psnr_y.item())
                psnr_u_list.append(psnr_u.item())
                psnr_v_list.append(psnr_v.item())
                rate_list.append(pixel_bit.item())
                test_loss.append(loss.item())

                if self.decoded:
                    y_hat, u_hat, v_hat = outputs
                    y_hat = torch.squeeze(y_hat,dim=0).cpu().numpy()*255.0
                    u_hat = torch.squeeze(u_hat,dim=0).cpu().numpy()*255.0
                    v_hat = torch.squeeze(v_hat,dim=0).cpu().numpy()*255.0
                    # y_new = y_comp.cpu().numpy()
                    # y_new = y_new*255.0
                    # mse = np.mean((y_hat-y_new)**2)
                    # psnr = 10 * np.log10(255 ** 2 / mse)
                    # print(psnr)
                    # assert False

                    y_hat = y_hat.astype(np.uint8)
                    u_hat = u_hat.astype(np.uint8)
                    v_hat = v_hat.astype(np.uint8)
                    print(y_hat.shape)
                    print(u_hat.shape)
                    print(v_hat.shape)
                    u_new = u_comp.cpu().numpy()
                    u_new = u_new*255.0
                    u_new = u_new.astype(np.uint8)
                    
                    # print(y_new)
                    #print(y_hat)
                    
                    
                    print(y_hat.shape,u_hat.shape)
                    
                    
                    np.savetxt("./original.txt", np.squeeze(u_new.astype(np.float32)))
                    np.savetxt("./recon.txt", np.squeeze(u_hat.astype(np.float32)))
                    np.savetxt('./diff.txt',np.abs(np.squeeze(u_new.astype(np.float32))-np.squeeze(u_hat.astype(np.float32))))
                    diff = np.squeeze(u_new.astype(np.float32))-np.squeeze(u_hat.astype(np.float32))
                    # print(np.max(diff))
                    # assert False
                    
                    
                    yuv_export(self.outputs_dir / '{}.yuv'.format(index),y_hat,u_hat,v_hat,1)
                    
                    
                    
                # original
                # if self.decoded:
                #     recon = outputs.squeeze().permute(1, 2, 0)
                #     recon = recon.cpu().numpy().astype(np.uint8)
                #     imsave(self.outputs_dir / "{}.png".format(index), recon)

        psnr_y_value = np.mean(psnr_y_list)
        psnr_u_value = np.mean(psnr_u_list)
        psnr_v_value = np.mean(psnr_v_list)
        rate_value = np.mean(rate_list)
        test_loss_value = np.mean(test_loss)
        # hm_diff_value = psnr_value - self.anchor(rate_value, "HM420")
        # vtm_diff_value = psnr_value - self.anchor(rate_value, "VTM420")

        return {"psnr_y": psnr_y_value, "bpp":rate_value, "psnr_u":psnr_u_value, "psnr_v":psnr_v_value, "loss":test_loss_value}


class TestingProc2(BasicOps):
    def __init__(self, options):
        super().__init__(options)
        # Set the configurations
        self.decoded = False

        # Define the important tools
        self.test_loader = None
        self._init_dataset()

        self.anchor = None
        self._init_tools()

        # Load the data
        # duan modify 2021.2.12 这个地方这个时候不需要load 因为self.factor和self.network都是赋值进去的 所以不需要
        # 原先那种方式需要是因为这个地方是test阶段 要看才需要
        self.factor = self._load()
        

        # Create the output directory
        if self.decoded:
            self.outputs_dir = self.log / "reconstruction"
            if not self.outputs_dir.exists():
                self.outputs_dir.mkdir()
            else:
                print("Directory has existed!")
                


    def _init_dataset(self):
        test_dataset = TestData()
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

    def _init_tools(self):
        self.anchor = LineFit()

    def _save(self, iteration, new_dict):
        ckpt_file = self.log / "checkpoint-{}.pth".format(iteration)
        checkpoint = {
            "factor": self.factor,
            "iteration": iteration,
            "network": new_dict,
            # 'loop_filter': self.loop_filter.state_dict(),
            # "main_optimizer": self.main_optimizer.state_dict(),
        }
        self.checkpoints.save(ckpt_file, checkpoint)

    @staticmethod
    def _norm(x):
        return (x * 255.0).clamp(0, 255).round_()

    def run(self):
        # modify the parameter because the p
        # new_dict = OrderedDict()
        # for key in self.network.state_dict():
        #     if "decoder.attention2." in key:
        #         print(key)
        #         key_new = key[0:8] + "attention1" + key[18:]
        #         print(key_new)
        #         new_dict[key_new] = self.network.state_dict()[key]
        #         new_dict[key] = self.network.state_dict()[key]
        #     else:
        #         new_dict[key] = self.network.state_dict()[key]
    		    
        # self._save(393700, new_dict)
       
        # Begin the training process
        psnr_y_list = list()
        psnr_u_list = list()
        psnr_v_list = list()
        time_list = list()
        test_loss = list()
        rate_list = list()
        self.network.eval()
        self.loop_filter.eval()
        # self.loop_filter_U.eval()
        # self.loop_filter_V.eval()
        self.loop_filter_UV.eval()
        with torch.no_grad():
            for index, inputs in enumerate(self.test_loader):
                # Get the input blocks
                # y_comp, u_comp, v_comp = [item for item in inputs]
                # time1 = time.time()
                y_comp, u_comp, v_comp = [item.to(**g_types, non_blocking=True) for item in inputs]
                
                loss, mse, pixel_bit, outputs, mse_u, mse_v, cost_time = self.forward_one_pass(y_comp, u_comp, v_comp)
                # time2 = time.time()
                # print("delta time:", time2-time1)
                # print(mse, mse_u, mse_v)
                time_list.append(cost_time)
                psnr_y = 10 * torch.log10(255 ** 2 / mse)
                psnr_u = 10 * torch.log10(255 ** 2 / mse_u)
                psnr_v = 10 * torch.log10(255 ** 2 / mse_v)
                # print(psnr)
                psnr_y_list.append(psnr_y.item())
                print(psnr_y.item())
                psnr_u_list.append(psnr_u.item())
                psnr_v_list.append(psnr_v.item())
                rate_list.append(pixel_bit.item())
                test_loss.append(loss.item())
                # print(ms_ssim.item())

                if self.decoded:
                    y_hat, u_hat, v_hat = outputs
                    y_hat = torch.squeeze(y_hat,dim=0).cpu().numpy()*255.0
                    u_hat = torch.squeeze(u_hat,dim=0).cpu().numpy()*255.0
                    v_hat = torch.squeeze(v_hat,dim=0).cpu().numpy()*255.0
                    
                    # y_new = y_comp.cpu().numpy()
                    # y_new = y_new*255.0
                    # mse = np.mean((y_hat-y_new)**2)
                    # psnr = 10 * np.log10(255 ** 2 / mse)
                    # print(psnr)
                    # assert False
                    y_hat = np.clip(y_hat, 17.0, 235.0)
                    y_new = y_comp.cpu().numpy()*255.0
                    mse = np.mean((y_hat-y_new)**2)
                    psnr = 10 * np.log10(255 ** 2 / mse)
                    print(psnr)

                    y_hat = np.round(y_hat)
                    u_hat = np.round(u_hat)
                    v_hat = np.round(v_hat)

                    y_hat = y_hat.astype(np.uint8)
                    u_hat = u_hat.astype(np.uint8)
                    v_hat = v_hat.astype(np.uint8)

                    
                    # y_next = y_hat.astype(np.float32)
                    # mse = np.mean((y_next-y_new)**2)
                    # psnr = 10 * np.log10(255 ** 2 / mse)
                    # print(psnr)

                    print(y_hat.shape)
                    print(u_hat.shape)
                    print(v_hat.shape)
                    u_new = u_comp.cpu().numpy()
                    u_new = u_new*255.0
                    u_new = u_new.astype(np.uint8)
                    
                    # print(y_new)
                    #print(y_hat)
                    
                    
                    print(y_hat.shape,u_hat.shape)
                    
                    q = np.max(np.abs(np.squeeze(u_new.astype(np.float32))-np.squeeze(u_hat.astype(np.float32))))
                    print(q)
                    np.savetxt("./original.txt", np.squeeze(u_new.astype(np.float32)))
                    np.savetxt("./recon.txt", np.squeeze(u_hat.astype(np.float32)))
                    np.savetxt('./diff.txt',np.abs(np.squeeze(u_new.astype(np.float32))-np.squeeze(u_hat.astype(np.float32))))
                    diff = np.squeeze(u_new.astype(np.float32))-np.squeeze(u_hat.astype(np.float32))
                    # print(np.max(diff))
                    # assert False
                    
                    
                    yuv_export(self.outputs_dir / '{}.yuv'.format(index),y_hat,u_hat,v_hat,1)
                    # Y_,U_,V_ = yuv_import(self.outputs_dir / '{}.yuv'.format(index),1920,1080,1)
                    print("yes")
                    
                    

        psnr_y_value = np.mean(psnr_y_list)
        psnr_u_value = np.mean(psnr_u_list)
        psnr_v_value = np.mean(psnr_v_list)
        test_loss_value = np.mean(test_loss)
        rate_value = np.mean(rate_list)
        print("time is {0}\n".format(np.mean(time_list)))

        print("[Factor: {}] ".format(self.factor), end='\t')
        print("PSNR_Y: {:>5.2f} dB ".format(psnr_y_value), end='\t')
        print("PSNR_U: {:>5.2f} dB ".format(psnr_u_value), end='\t')
        print("PSNR_V: {:>5.2f} dB ".format(psnr_v_value), end='\t')
        print("Test loss: {:>2.6f} ".format(test_loss_value), end='\t')
        print("Rate: {:>6.4f} bpp ".format(rate_value), end='\n')
        

if __name__ == "__main__":
    if opts.mode == "train":
        obj = TrainingProc(opts)
        obj.run()
    elif opts.mode == "test":
        obj = TestingProc2(opts)
        obj.run()
    else:
        print(opts.mode)
        raise ValueError("Mode should be in [train|test].")
