"""General-purpose test script for Generative Shadow Synthesis Model (GSS).

This script is used to test the shadow synthesis model. It loads a saved model from '--checkpoints_dir'
and saves the results to '--results_dir'. The script supports testing on synthetic datasets and visualizing
the results in an HTML file.

Example:
    Test the GSS model:
        python test_GSS.py --name AISD_GSSNet --model test_GSS --dataset_mode Syn_unaligned --custom_dataset_size 5149 --syn_crop_size 256

Key features:
    - Hard-coded parameters for testing (e.g., batch size, no flipping, serial batches).
    - Saves results to an HTML file for visualization.
    - Supports testing on a specified number of images using '--num_test'.

See `options/base_options.py` and `options/test_options.py` for more test options.
"""
from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_syn_images

if __name__ == "__main__":
    opt = TestOptions().parse()  # Parse test options

    # Set experiment parameters
    opt.name = "AISD_GSSNet"  # Experiment name
    opt.model = "test_GSS"  # Model type
    opt.model_suffix = "_A"  # Model suffix
    opt.dataset_mode = "Syn_unaligned"  # Dataset mode

    # Hard-code testing parameters
    opt.num_threads = 0  # Test code only supports num_threads = 1
    opt.batch_size = 1  # Test code only supports batch_size = 1
    opt.serial_batches = True  # Disable data shuffling; use sequential batches
    opt.no_flip = True  # Disable image flipping for testing
    opt.display_id = -1  # Disable visdom display; results are saved to an HTML file

    # Create model and dataset
    model = create_model(opt)  # Initialize the model
    dataset = create_dataset(opt)  # Load the dataset
    model.setup(opt)  # Load model weights and configure settings

    # Create a webpage for saving results
    webpage = html.HTML(
        opt.GSS_results_dir,
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
        save_syn_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
            transfer2white=opt.transfer2white,
        )
    webpage.save()  # Save the HTML file