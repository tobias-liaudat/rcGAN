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
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tick


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    #TODO: Refactor config path
    with open('/home/jjwhit/rcGAN/configs/mass_map_8.yml', 'r') as f:
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
                zoom_startx = np.random.randint(120, 250)
                zoom_starty1 = np.random.randint(30, 80)
                zoom_starty2 = np.random.randint(260, 300)

                p = np.random.rand()
                zoom_starty = zoom_starty1
                if p <= 0.5:
                    zoom_starty = zoom_starty2

                zoom_length = 80

                x_coord = zoom_startx + zoom_length
                y_coords = [zoom_starty, zoom_starty + zoom_length]

                # Global recon, error, std
                nrow = 2
                ncol = 2

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.25, hspace=0.45,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax1 = plt.subplot(gs[0, 0])
                im1 = ax1.imshow(np_gt, cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt))
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_title("Truth", fontsize=7)

                ax2 = plt.subplot(gs[0, 1])
                im2 = ax2.imshow(np_avgs[method], cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt))
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.set_title("Reconstruction", fontsize=7)

                cbar1 = plt.colorbar(im2, ax=ax2, shrink=0.6)
                cbar1.ax.tick_params(labelsize=3)

                ax3 = plt.subplot(gs[1, 0])
                im3 = ax3.imshow(np.abs(np_avgs[method] - np_gt), cmap='jet', vmin=0,
                               vmax=0.4 *np.max(np.abs(np_avgs['mmGAN'] - np_gt)))
                ax3.set_xticklabels([])
                ax3.set_yticklabels([])
                ax3.set_xticks([])
                ax3.set_yticks([])
                ax3.set_title("Absolute Error", fontsize=7)

                cbar2 = plt.colorbar(im3, ax=ax3, shrink=0.6)
                cbar2.ax.tick_params(labelsize=3)

                ax4 = plt.subplot(gs[1, 1])
                im4 =ax4.imshow(np_stds[method], cmap='viridis', vmin=0, vmax=0.4 * np.max(np_stds['mmGAN']))
                ax4.set_xticklabels([])
                ax4.set_yticklabels([])
                ax4.set_xticks([])
                ax4.set_yticks([])
                ax4.set_title("Std. Dev.", fontsize=7)

                cbar3 = plt.colorbar(im4, ax=ax4, shrink=0.6)
                cbar3.ax.tick_params(labelsize=3)

                #plt.savefig(f'/share/gpu0/jjwhit/test_figures_1/test_fig_1_avg_err_std_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.savefig(f'/home/jjwhit/rcGAN/jobs/validate_test/avg_err_std_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)


                # New figure
                cbar_font_size = 14
                title_fonts = 20
                fig = plt.figure(figsize=(12,12))

                plt.subplot(221)
                ax = plt.gca()
                ax.set_title("Truth", fontsize=title_fonts)
                im = ax.imshow(np_gt, cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt))
                ax.set_xticks([]);ax.set_yticks([])

                plt.subplot(222)
                ax = plt.gca()
                ax.set_title("Reconstruction", fontsize=title_fonts)
                im = ax.imshow(np_avgs[method], cmap='inferno', vmin=0, vmax=0.4 * np.max(np_gt))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
                cbar.ax.tick_params(labelsize=cbar_font_size)
                ax.set_xticks([]);ax.set_yticks([])                

                plt.subplot(223)
                ax = plt.gca()
                ax.set_title("Absolute Error", fontsize=title_fonts)
                im = ax.imshow(
                    np.abs(np_avgs[method] - np_gt),
                    cmap='jet',
                    vmin=0,
                    vmax=0.2 *np.max(np.abs(np_avgs['mmGAN'] - np_gt))
                )
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
                cbar.ax.tick_params(labelsize=cbar_font_size)
                ax.set_xticks([]);ax.set_yticks([])

                plt.subplot(224)
                ax = plt.gca()
                ax.set_title("Std. Dev.", fontsize=title_fonts)
                im = ax.imshow(
                    np_stds[method], cmap='viridis', vmin=0, vmax=0.4 * np.max(np_stds['mmGAN'])
                )
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
                cbar.ax.tick_params(labelsize=cbar_font_size)
                ax.set_xticks([]);ax.set_yticks([])

                plt.savefig(f'/home/jjwhit/rcGAN/jobs/validate_test/avg_err_std_{fig_count}_v2.png', bbox_inches='tight', dpi=300)
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

                #HERE
                for samp in range(2):
                    ax = plt.subplot(gs[1, samp])
                    ax.imshow(np_samps[method][samp][zoom_starty:zoom_starty + zoom_length,
                              zoom_startx:zoom_startx + zoom_length], cmap='inferno', vmin=0,
                              vmax=0.5 * np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}', fontsize=8)

                ax = plt.subplot(gs[2, 0])
                ax.imshow(
                    np_avgs[method][zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Reconstruction', fontsize=8)

                # ax = plt.subplot(gs[0, 5])
                # avg = np.zeros((384, 384))
                # for l in range(8):
                #     avg += np_samps[method][l]
                # #avg = avg / 4
                # avg = avg / 8

                # ax.imshow(
                #     avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                #     cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                # ax.set_xticklabels([])
                # ax.set_yticklabels([])
                # ax.set_xticks([])
                # ax.set_yticks([]) 
                # ax.set_title('8-Avg.') #Originally 4-avg

                # ax = plt.subplot(gs[0, 6])
                # avg = np.zeros((384, 384))
                # for l in range(4):
                #     avg += np_samps[method][l]

                # # avg = avg / 2
                # avg = avg / 4
                # ax.imshow(
                #     avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                #     cmap='inferno', vmin=0, vmax=0.5 * np.max(np_gt))
                # ax.set_xticklabels([])
                # ax.set_yticklabels([])
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.set_title('4-Avg.') #Originally 2-avg

                ax = plt.subplot(gs[2, 1])
                ax.imshow(np.abs(np_avgs[method][zoom_starty:zoom_starty + zoom_length,    
                          zoom_startx:zoom_startx + zoom_length] - np_gt[zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length]), cmap='jet', vmin=0,
                               vmax=0.25 *np.max(np.abs(np_avgs['mmGAN'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Absolute Error", fontsize=8)

                ax = plt.subplot(gs[3, 0])
                ax.imshow(np_stds[method][zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length], cmap='viridis', vmin=0,
                          vmax=0.5 *np.max(np_stds['mmGAN']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Std. Dev.', fontsize=8)

                #plt.savefig(f'/share/gpu0/jjwhit/test_figures_1/zoomed_avg_1_samps_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.savefig(f'/home/jjwhit/rcGAN/jobs/validate_test/zoomed_avg_samps_{fig_count}.png', bbox_inches='tight', dpi=300)
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
                avg = np.zeros((384, 384))
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
                ax.set_title('8-Avg.')

                
                ax = plt.subplot(gs[2, 0])
                avg = np.zeros((384, 384))
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
                ax.set_title('4-Avg.')


                ax = plt.subplot(gs[2, 1])
                avg = np.zeros((384, 384))
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
                ax.set_title('2-Avg.')

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
                plt.savefig(f'/home/jjwhit/rcGAN/jobs/validate_test/avg_err_std_{fig_count}_v2.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                if fig_count == args.num_figs:
                    sys.exit()
                fig_count += 1
