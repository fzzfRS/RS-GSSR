import ntpath
import os
import sys
import time
from subprocess import PIPE, Popen

import numpy as np

from . import html, util

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = "%s_%s.png" % (name, label)
        image_name = "%s.tif" % (name)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


def save_syn_images(
    webpage, visuals, image_path, aspect_ratio=1.0, width=256, transfer2white=True
):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    root_dir = webpage.get_image_dir()[:-7]
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():

        im = util.tensor2im(im_data, np.uint8, transfer2white)
        image_name = short_path
        target_dir = os.path.join(root_dir, label)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        save_path = os.path.join(target_dir, image_name)

        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if (
            self.display_id > 0
        ):  # connect to a visdom server given <display_port> and <display_server>
            import visdom

            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(
                server=opt.display_server, port=opt.display_port, env=opt.display_env
            )
            # zxr修改
            # if not self.vis.check_connection():
            self.create_visdom_connections()

        if (
            self.use_html
        ):  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, "web")
            self.img_dir = os.path.join(self.web_dir, "images")
            print("create web directory %s..." % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                "================ Training Loss (%s) ================\n" % now
            )

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port >"""
        cmd = sys.executable + " -m visdom.server -p %d &>/dev/null &" % self.port
        print("\n\nCould not connect to Visdom server. \n Trying to start a server....")
        print("Command: %s" % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:  # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (
                    w,
                    h,
                )  # create a table css
                # create a table of images.
                title = self.name
                label_html = ""
                label_html_row = ""
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += "<td>%s</td>" % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += "<tr>%s</tr>" % label_html_row
                        label_html_row = ""
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += "<td></td>"
                    idx += 1
                if label_html_row != "":
                    label_html += "<tr>%s</tr>" % label_html_row
                try:
                    self.vis.images(
                        images,
                        nrow=ncols,
                        win=self.display_id + 1,
                        padding=2,
                        opts=dict(title=title + " images"),
                    )
                    label_html = "<table>%s</table>" % label_html
                    self.vis.text(
                        table_css + label_html,
                        win=self.display_id + 2,
                        opts=dict(title=title + " labels"),
                    )
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:  # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(
                            image_numpy.transpose([2, 0, 1]),
                            opts=dict(title=label),
                            win=self.display_id + idx,
                        )
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (
            save_result or not self.saved
        ):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(
                    self.img_dir, "epoch%.3d_%s.png" % (epoch, label)
                )
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(
                self.web_dir, "Experiment name = %s" % self.name, refresh=1
            )
            for n in range(epoch, 0, -1):
                webpage.add_header("epoch [%d]" % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = "epoch%.3d_%s.png" % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, "plot_data"):
            self.plot_data = {"X": [], "Y": [], "legend": list(losses.keys())}
        self.plot_data["X"].append(epoch + counter_ratio)
        self.plot_data["Y"].append([losses[k] for k in self.plot_data["legend"]])
        try:
            self.vis.line(
                X=np.stack(
                    [np.array(self.plot_data["X"])] * len(self.plot_data["legend"]), 1
                ),
                Y=np.array(self.plot_data["Y"]),
                opts={
                    "title": self.name + " loss over time",
                    "legend": self.plot_data["legend"],
                    "xlabel": "epoch",
                    "ylabel": "loss",
                },
                win=self.display_id,
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = "(epoch: %d, iters: %d, time: %.3f, data: %.3f) " % (
            epoch,
            iters,
            t_comp,
            t_data,
        )
        for k, v in losses.items():
            message += "%s: %.3f " % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)  # save the message
