"""General-purpose training script for Generative Shadow Synthesis Model (GSS).

This script is used to train the shadow synthesis model. It supports various models and datasets.
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

The script creates the model, dataset, and visualizer based on the provided options.
It performs standard network training, visualizes/saves images, prints/saves loss plots, and saves models.
The script also supports resuming training using '--continue_train'.

Example:
    Train the GSS model:
        python train_GSS.py --dataroot ./datasets/shadow_data --name AISD_GSSNet --model GSS --dataset_mode Syn_unaligned --custom_dataset_size 1000 --syn_crop_size 256

See `options/base_options.py` and `options/train_options.py` for more training options.
"""
import time

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer

if __name__ == "__main__":
    opt = TrainOptions().parse()  # Parse training options

    opt.name = "AISD_GSSNet"  # Experiment name
    opt.netG = "unet_256"  # Generator network type
    opt.model = "GSS"  # Model type
    opt.dataset_mode = "Syn_unaligned"  # Dataset mode


    # Create dataset and get the number of images
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print("The number of training images = %d" % dataset_size)

    # Create model and visualizer
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    total_iters = 0  # Total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        # Outer loop for epochs
        epoch_start_time = time.time()  # Timer for the entire epoch
        iter_data_time = time.time()  # Timer for data loading per iteration
        epoch_iter = 0  # Number of iterations in the current epoch

        for i, data in enumerate(dataset):  # Inner loop for iterations
            iter_start_time = time.time()  # Timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)  # Unpack data and apply preprocessing
            model.optimize_parameters()  # Calculate losses, gradients, and update weights

            if total_iters % opt.display_freq == 0:
                # Display images and save results to an HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result
                )

            if total_iters % opt.print_freq == 0:
                # Print training losses and save logs
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data
                )
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            # Save the model at the end of the epoch
            print(
                "Saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
        )
        model.update_learning_rate()  # Update learning rates at the end of the epoch