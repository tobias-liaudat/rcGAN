import torch
import yaml
import types
import json

import numpy as np
import matplotlib.patches as patches

import sys
sys.path.append('/home/jjwhit/rcGAN/')

from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.mmGAN import mmGAN
from utils.mri.math import tensor_to_complex_np
from scipy import ndimage
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tick


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    config_path = args.config

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MMDataModule(cfg)
    fig_count = 1
    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        mmGAN_model = mmGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')

        mmGAN_model.cuda()

        mmGAN_model.eval()

        for i, data in enumerate(test_loader):
            y, x, mean, std = data
            y = y.cuda()
            x = x.cuda()
            mean = mean.cuda()
            std = std.cuda()

            gens_mmGAN = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size, 2)).cuda()

            for z in range(cfg.num_z_test):
                gens_mmGAN[:, z, :, :, :] = mmGAN_model.reformat(mmGAN_model.forward(y))

            avg_mmGAN = torch.mean(gens_mmGAN, dim=1)

            gt = mmGAN_model.reformat(x)
            zfr = mmGAN_model.reformat(y)

            for j in range(y.size(0)):
                np_avgs = {
                    'mmGAN': None,
                }

                np_samps = {
                    'mmGAN': [],
                }

                np_stds = {
                    'mmGAN': None,
                }

                np_gt = None

                np_gt = ndimage.rotate(
                    torch.tensor(tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)
                np_zfr = ndimage.rotate(
                    torch.tensor(tensor_to_complex_np((zfr[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)

                np_avgs['mmGAN'] = ndimage.rotate(
                    torch.tensor(tensor_to_complex_np((avg_mmGAN[j] * std[j] + mean[j]).cpu())).abs().numpy(),
                    180)

                for z in range(cfg.num_z_test):
                    np_samps['mmGAN'].append(ndimage.rotate(torch.tensor(
                        tensor_to_complex_np((gens_mmGAN[j, z] * std[j] + mean[j]).cpu())).abs().numpy(), 180))

                np_stds['mmGAN'] = np.std(np.stack(np_samps['mmGAN']), axis=0)

                method = 'mmGAN'
                zoom_length = 60  # Adjust this value based on your preference
                margin = 10  # Adjust this value to set the margin

                # Ensure the square is not touching the edge
                zoom_startx = np.random.randint(margin, cfg.im_size - zoom_length - margin)
                zoom_starty1 = np.random.randint(margin, int(cfg.im_size / 2) - zoom_length - margin)
                zoom_starty2 = np.random.randint(int(cfg.im_size / 2) + margin, cfg.im_size - zoom_length - margin)

                p = np.random.rand()
                zoom_starty = zoom_starty1 if p <= 0.5 else zoom_starty2

                x_coord = zoom_startx + zoom_length
                y_coords = [zoom_starty, zoom_starty + zoom_length]



                # Global recon, error, std
                nrow = 2
                ncol = 2

                #fig = plt.figure(figsize=(ncol + 1, nrow + 1))
                fig, axes = plt.subplots(nrow, ncol, figsize=(8,8), constrained_layout=True)

                axes[0,0].imshow(np_gt, aspect='auto', cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt), origin='lower')
                axes[0,0].set_title('Truth')
                axes[0,0].set_xticklabels([])
                axes[0,0].set_yticklabels([])
                axes[0,0].set_xticks([])
                axes[0,0].set_yticks([])


                im1 = axes[0,1].imshow(np_avgs[method], cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt), origin='lower')
                axes[0,1].set_title('Reconstruction')
                axes[0,1].set_xticklabels([])
                axes[0,1].set_yticklabels([])
                axes[0,1].set_xticks([])
                axes[0,1].set_yticks([])
                plt.colorbar(im1, ax=axes[0,1], shrink=0.8)

                im2 = axes[1,0].imshow(np.abs(np_avgs[method] - np_gt), cmap='jet', vmin=0, vmax=0.4 *np.max(np.abs(np_avgs['mmGAN'] - np_gt)), origin='lower')
                axes[1,0].set_title('Absolute Error')
                axes[1,0].set_xticklabels([])
                axes[1,0].set_yticklabels([])
                axes[1,0].set_xticks([])
                axes[1,0].set_yticks([])
                plt.colorbar(im2, ax=axes[1,0], shrink=0.8)

                im3 = axes[1,1].imshow(np_stds[method], cmap='viridis', vmin=0, vmax=0.4 * np.max(np_stds['mmGAN']), origin='lower')
                axes[1,1].set_title('Std. Dev.')
                axes[1,1].set_xticklabels([])
                axes[1,1].set_yticklabels([])
                axes[1,1].set_xticks([])
                axes[1,1].set_yticks([])
                plt.colorbar(im2, ax=axes[1,1], shrink=0.8)

                axes[0,0].set_aspect('equal')

                plt.savefig(f'/share/gpu0/jjwhit/plots/cosmos_training_plots/square_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                # #New figure
                # cbar_font_size = 14
                # title_fonts = 20
                # #fig = plt.figure(figsize=(12,12))
                # fig = plt.subplots(nrow, ncol, figsize=(2,2), constrained_layout=True)

                # plt.subplot(221)
                # ax = plt.gca()
                # ax.set_title("Truth", fontsize=title_fonts)
                # im = ax.imshow(np_gt, cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt))
                # ax.set_xticks([]);ax.set_yticks([])

                # plt.subplot(222)
                # ax = plt.gca()
                # ax.set_title("Reconstruction", fontsize=title_fonts)
                # im = ax.imshow(np_avgs[method], cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt))
                # cbar = plt.colorbar(im, ax=ax)
                # # divider = make_axes_locatable(ax)
                # # cax = divider.append_axes('right', size='5%', pad=0.05)
                # # cbar = fig.colorbar(im, cax=cax)
                # # cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
                # # cbar.ax.tick_params(labelsize=cbar_font_size)
                # ax.set_xticks([]);ax.set_yticks([])                

                # plt.subplot(223)
                # ax = plt.gca()
                # ax.set_title("Absolute Error", fontsize=title_fonts)
                # im = ax.imshow(
                #     np.abs(np_avgs[method] - np_gt),
                #     cmap='jet',
                #     vmin=0,
                #     vmax=0.2 *np.max(np.abs(np_avgs['mmGAN'] - np_gt))
                # )
                # cbar = plt.colorbar(im, ax=ax)
                # # divider = make_axes_locatable(ax)
                # # cax = divider.append_axes('right', size='5%', pad=0.05)
                # # cbar = fig.colorbar(im, cax=cax)
                # # cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
                # # cbar.ax.tick_params(labelsize=cbar_font_size)
                # ax.set_xticks([]);ax.set_yticks([])

                # plt.subplot(224)
                # ax = plt.gca()
                # ax.set_title("Std. Dev.", fontsize=title_fonts)
                # im = ax.imshow(
                #     np_stds[method], cmap='viridis', vmin=0, vmax=0.4 * np.max(np_stds['mmGAN'])
                # )
                # cbar = plt.colorbar(im, ax=ax)
                # # divider = make_axes_locatable(ax)
                # # cax = divider.append_axes('right', size='5%', pad=0.05)
                # # cbar = fig.colorbar(im, cax=cax)
                # # cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
                # # cbar.ax.tick_params(labelsize=cbar_font_size)
                # ax.set_xticks([]);ax.set_yticks([])

                # plt.savefig(f'/share/gpu0/jjwhit/plots/cosmos_training_plots/avg_err_std_{fig_count}_v2.png', bbox_inches='tight', dpi=300)
                # plt.close(fig)


                # New figure
                cbar_font_size = 14
                title_fonts = 20
                fig, axs = plt.subplots(nrow, ncol, figsize=(8, 8), constrained_layout=True)

                axs = axs.flatten()  # Flatten the 2D array of Axes

                axs[0].set_title("Truth", fontsize=title_fonts)
                im = axs[0].imshow(np_gt, cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt))
                axs[0].set_xticks([]); axs[0].set_yticks([])

                axs[1].set_title("Reconstruction", fontsize=title_fonts)
                im = axs[1].imshow(np_avgs[method], cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt))
                cbar = plt.colorbar(im, ax=axs[1])
                axs[1].set_xticks([]); axs[1].set_yticks([])

                axs[2].set_title("Absolute Error", fontsize=title_fonts)
                im = axs[2].imshow(
                    np.abs(np_avgs[method] - np_gt),
                    cmap='jet',
                    vmin=0,
                    vmax=0.2 * np.max(np.abs(np_avgs['mmGAN'] - np_gt))
                )
                cbar = plt.colorbar(im, ax=axs[2])
                axs[2].set_xticks([]); axs[2].set_yticks([])

                axs[3].set_title("Std. Dev.", fontsize=title_fonts)
                im = axs[3].imshow(
                    np_stds[method], cmap='viridis', vmin=0, vmax=0.4 * np.max(np_stds['mmGAN'])
                )
                cbar = plt.colorbar(im, ax=axs[3])
                axs[3].set_xticks([]); axs[3].set_yticks([])

                plt.savefig(f'/share/gpu0/jjwhit/plots/cosmos_training_plots/avg_err_std_{fig_count}_v2.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                # Plot 2
                nrow = 4
                ncol = 2
                
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt, cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth', fontsize=7)

                ax1 = ax

                rect = patches.Rectangle((zoom_startx, zoom_starty), zoom_length, zoom_length, linewidth=1,
                                         edgecolor='r',
                                         facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                          cmap='inferno',
                          vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth', fontsize=7)

                connection_path_1 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty],
                                                            [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty + zoom_length], [0, zoom_length],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                #HERE
                for samp in range(1):
                    ax = plt.subplot(gs[1, samp])
                    ax.imshow(np_samps[method][samp][zoom_starty:zoom_starty + zoom_length,
                              zoom_startx:zoom_startx + zoom_length], cmap='inferno', vmin=0,
                              vmax=0.5 * np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}', fontsize=7)

                ax = plt.subplot(gs[1, 1])
                ax.imshow(
                    np_avgs[method][zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Reconstruction', fontsize=7)

                ax = plt.subplot(gs[2, 0])
                ax.imshow(np.abs(np_avgs[method][zoom_starty:zoom_starty + zoom_length,    
                          zoom_startx:zoom_startx + zoom_length] - np_gt[zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length]), cmap='jet', vmin=0,
                               vmax=0.25 *np.max(np.abs(np_avgs['mmGAN'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Absolute Error", fontsize=7)

                ax = plt.subplot(gs[2, 1])
                ax.imshow(np_stds[method][zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length], cmap='viridis', vmin=0,
                          vmax=0.5 *np.max(np_stds['mmGAN']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Std. Dev.', fontsize=7)

                plt.savefig(f'/share/gpu0/jjwhit/plots/cosmos_training_plots/zoomed_overview_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                #Plot number 3

                nrow = 4
                ncol = 2

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt, cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth', fontsize=8)

                ax1 = ax

                rect = patches.Rectangle((zoom_startx, zoom_starty), zoom_length, zoom_length, linewidth=1,
                                         edgecolor='r',
                                         facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                          cmap='inferno',
                          vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth', fontsize=8)

                connection_path_1 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty],
                                                            [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty + zoom_length], [0, zoom_length],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                ax = plt.subplot(gs[1, 0])
                ax.imshow(
                    np_avgs[method][zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Reconstruction', fontsize=8)

                ax = plt.subplot(gs[1, 1])
                avg = np.zeros((cfg.im_size, cfg.im_size))
                for l in range(8):
                    avg += np_samps[method][l]
                avg = avg / 8

                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([]) 
                ax.set_title('8-Avg.', fontsize=8)

                
                ax = plt.subplot(gs[2, 0])
                avg = np.zeros((cfg.im_size, cfg.im_size))
                for l in range(4):
                    avg += np_samps[method][l]

                avg = avg / 4
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('4-Avg.', fontsize=8)


                ax = plt.subplot(gs[2, 1])
                avg = np.zeros((cfg.im_size, cfg.im_size))
                for l in range(2):
                    avg += np_samps[method][l]

                avg = avg / 2
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('2-Avg.', fontsize=8)

                for samp in range(1):
                    ax = plt.subplot(gs[3, 0])
                    ax.imshow(np_samps[method][samp][zoom_starty:zoom_starty + zoom_length,
                                zoom_startx:zoom_startx + zoom_length], cmap='inferno', vmin=0,
                                vmax=0.5 * np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}', fontsize=8)

                ax = plt.subplot(gs[3, 1])
                ax.imshow(np_stds[method][zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length], cmap='viridis', vmin=0,
                          vmax=0.5 *np.max(np_stds['mmGAN']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Std. Dev.', fontsize=8)                

                #plt.savefig(f'/share/gpu0/jjwhit/test_figures_1/test_fig_1_avg_err_std_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.savefig(f'/share/gpu0/jjwhit/plots/cosmos_training_plots/avg_err_std_{fig_count}_v2.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                nrow = 3
                ncol = 2
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                
                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt, cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth', fontsize=8)

                ax1 = ax

                rect = patches.Rectangle((zoom_startx, zoom_starty), zoom_length, zoom_length, linewidth=1,
                                         edgecolor='r',
                                         facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                          cmap='inferno',
                          vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth', fontsize=8)

                connection_path_1 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty],
                                                            [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty + zoom_length], [0, zoom_length],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                for samp in range(2):
                    ax = plt.subplot(gs[1, samp])
                    ax.imshow(np_samps[method][samp][zoom_starty:zoom_starty + zoom_length,
                            zoom_startx:zoom_startx + zoom_length], cmap='inferno', vmin=0,
                            vmax=0.5 * np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}', fontsize=7)
                
                for samp in range(2):
                    ax = plt.subplot(gs[2, samp])
                    ax.imshow(np_samps[method][samp+2][zoom_starty:zoom_starty + zoom_length,
                            zoom_startx:zoom_startx + zoom_length], cmap='inferno', vmin=0,
                            vmax=0.5 * np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 3}', fontsize=7)
                

                plt.savefig(f'/share/gpu0/jjwhit/plots/cosmos_training_plots/diversity_{fig_count}_v2.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                nrow = 4
                ncol = 2
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.25,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                          cmap='inferno',
                          vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth', fontsize=8)

                ax1 = ax

                ax = plt.subplot(gs[0, 1])
                avg = np.zeros((cfg.im_size,cfg.im_size))
                for l in range(2):
                    avg += np_samps[method][l]

                avg = avg / 2
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('2-Avg.', fontsize=8)


                ax = plt.subplot(gs[1, 0])
                avg = np.zeros((cfg.im_size,cfg.im_size))
                for l in range(4):
                    avg += np_samps[method][l]

                avg = avg / 4
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('4-Avg.', fontsize=8)


                ax = plt.subplot(gs[1, 1])
                avg = np.zeros((cfg.im_size,cfg.im_size))
                for l in range(8):
                    avg += np_samps[method][l]

                avg = avg / 8
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('8-Avg.', fontsize=8)


                ax = plt.subplot(gs[2, 0])
                avg = np.zeros((cfg.im_size,cfg.im_size))
                for l in range(16):
                    avg += np_samps[method][l]

                avg = avg / 16
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('16-Avg.', fontsize=8)

                ax = plt.subplot(gs[2, 1])
                avg = np.zeros((cfg.im_size,cfg.im_size))
                for l in range(32):
                    avg += np_samps[method][l]

                avg = avg / 32
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('32-Avg.', fontsize=8)

                plt.savefig(f'/share/gpu0/jjwhit/plots/cosmos_training_plots/P_ascent_zoom_{fig_count}_v2.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                if fig_count == args.num_figs:
                    sys.exit()
                fig_count += 1
