# Python
import torch

from . import networks
from .base_model import BaseModel


class GSRModel(BaseModel):
    """
    This class implements the Generative Shadow Removal (GSR) model for shadow removal tasks.

    The model training requires the '--dataset_mode HEI_aligned' dataset.
    By default, it uses a '--netG UnetFusion' generator and a '--netD basic' discriminator.
    The model supports GAN loss, L1 loss, VGG loss, and color ratio loss for shadow removal.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options and rewrite default values for existing options.

        Args:
            parser: Original option parser.
            is_train (bool): Whether training phase or test phase.

        Returns:
            argparse.ArgumentParser: The modified parser with additional options.
        """
        if is_train:
            parser.add_argument("--lambda_L1", type=float, default=80.0, help="Weight for L1 loss.")
            parser.add_argument("--lambda_VGG", type=float, default=7.0, help="Weight for VGG loss.")
            parser.add_argument("--lambda_Color", type=float, default=200.0, help="Weight for color ratio loss.")
        return parser

    def __init__(self, opt):
        """Initialize the GSR model.

        Args:
            opt: Options containing model configurations.
        """
        super().__init__(opt)

        # Define loss names for training
        self.loss_names = [
            "G", "G_Syn_L1", "G_GAN", "G_Syn_VGG", "G_Syn_Color", "D_real", "D_fake"
        ]

        # Define visualizations for synthetic shadow removal
        self.visual_names = [
            "Syn_shadow", "Syn_HEI", "out_Synshadowfree", "Syn_shadowfree", "Syn_mask", "Syn_edge"
        ]

        # Define model names for saving/loading
        self.model_names = ["G", "D"] if self.isTrain else ["G"]

        # Define networks (generator and discriminator)
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids
        )

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc + opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids
            )

            # Define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionColor = networks.ColorRatioLoss().to(self.device)
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary preprocessing.

        Args:
            input (dict): Contains the data and metadata information.
        """
        self.Syn_shadow = input["umbra"].to(self.device)
        self.Syn_shadowfree = input["shadowfree"].to(self.device)
        self.Syn_HEI = input["de"].to(self.device)
        self.Syn_mask = input["umbra_mask"].to(self.device)
        self.Syn_edge = input["edge_mask"].to(self.device)
        self.image_paths = input["A_paths"]

    def forward(self):
        """Run forward pass to compute shadow-free images."""
        self.out_Synshadowfree = self.netG(
            self.Syn_shadow, self.Syn_HEI, self.Syn_mask, self.Syn_edge
        )

    def backward_G(self):
        """Calculate losses for the generator."""
        # GAN loss
        fake_AB = torch.cat((self.Syn_shadow, self.out_Synshadowfree), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1 loss
        self.loss_G_Syn_L1 = self.criterionL1(self.out_Synshadowfree, self.Syn_shadowfree) * self.opt.lambda_L1

        # VGG loss
        self.loss_G_Syn_VGG = self.criterionVGG(self.out_Synshadowfree, self.Syn_shadowfree) * self.opt.lambda_VGG

        # Color ratio loss
        self.loss_G_Syn_Color = self.criterionColor(self.out_Synshadowfree, self.Syn_shadowfree) * self.opt.lambda_Color

        # Combined loss
        self.loss_G = self.loss_G_Syn_L1 + self.loss_G_GAN + self.loss_G_Syn_VGG + self.loss_G_Syn_Color
        self.loss_G.backward()

    def backward_D(self):
        """Calculate losses for the discriminator."""
        # Fake loss
        fake_AB = torch.cat((self.Syn_shadow, self.out_Synshadowfree), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real loss
        real_AB = torch.cat((self.Syn_shadow, self.Syn_shadowfree), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def optimize_parameters(self):
        """Optimize parameters for both generator and discriminator."""
        self.forward()  # Compute fake images
        # Update discriminator
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Update generator
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()