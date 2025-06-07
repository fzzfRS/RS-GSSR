"""Test script for Generative Shadow Removal (GSR) model.

This script is used to test the shadow removal model. It loads a saved model from '--checkpoints_dir'
and saves the results to '--results_dir'. The script supports testing on synthetic datasets and visualizing
the results in an HTML file.

Example:
    Test the GSR model:
        python test_GSR.py --dataroot ./datasets/shadow_data --name AISD_GSR -- dataroot your dataset --model test_GSR --dataset_mode HEI_single

Key features:
    - Hard-coded parameters for testing (e.g., batch size, no flipping, serial batches).
    - Saves results to an HTML file for visualization.
    - Supports testing on a specified number of images using '--num_test'.

See `options/base_options.py` and `options/test_options.py` for more test options.
"""
import os

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images

if __name__ == "__main__":
    opt = TestOptions().parse()  # Parse test options

    # Set experiment parameters
    opt.dataset_mode = "HEI_single"  # Dataset mode for testing
    opt.netG = "UnetFusion"  # Generator network type
    opt.model = "test_GSR"  # Model type

    # Hard-code testing parameters
    opt.num_threads = 0  # Test code only supports num_threads = 1
    opt.batch_size = 1  # Test code only supports batch_size = 1
    opt.serial_batches = True  # Disable data shuffling; use sequential batches
    opt.no_flip = True  # Disable image flipping for testing
    opt.display_id = -1  # Disable visdom display; results are saved to an HTML file

    # Create model and dataset
    dataset = create_dataset(opt)  # Load the dataset
    model = create_model(opt)  # Initialize the model
    model.setup(opt)  # Load model weights and configure settings

    # Create a webpage for saving results
    web_dir = os.path.join(
        opt.GSR_results_dir, opt.name, "%s_%s" % (opt.phase, opt.epoch)
    )
    img_dir = os.path.join(web_dir, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )

    # Set model to evaluation mode
    model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # Limit the number of test images
            break
        model.set_input(data)  # Unpack data from the dataloader
        model.test()  # Run inference
        visuals = model.get_current_visuals()  # Get generated images
        img_path = model.get_image_paths()  # Get image paths
        if i % 1 == 0:  # Save images to the HTML file
            print("Processing (%04d)-th image... %s" % (i, img_path))
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
        )
    webpage.save()  # Save the HTML file