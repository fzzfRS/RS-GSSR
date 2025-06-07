import itertools
import torch
from util.image_pool import ImagePool
from . import networks
from .base_model import BaseModel


class GSSModel(BaseModel):
    """
    This class implements the Generative Shadow Synthesis (GSS) model for shadow synthesis tasks.

    The model training requires the '--dataset_mode Syn_unaligned' dataset.
    By default, it uses a '--netG unet_256' generator and a '--netD basic' discriminator.
    The model supports L1 loss for shadow consistency and cycle consistency losses for domain translation.
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
        parser.set_defaults(no_dropout=True)  # Default: no dropout
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, help="Weight for cycle loss (A -> B -> A).")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="Weight for cycle loss (B -> A -> B).")
            parser.add_argument("--lambda_L1", type=float, default=10.0, help="Weight for shadow consistency loss.")
            parser.add_argument("--lambda_identity", type=float, default=0.5, help="Weight for identity mapping loss.")

        return parser

    def __init__(self, opt):
        """Initialize the GSS model.

        Args:
            opt: Options containing model configurations.
        """
        super().__init__(opt)

        # Define loss names for training
        self.loss_names = [
            "D_A", "G_A", "cycle_A", "idt_A", "G_A_black_consis",
            "D_B", "G_B", "cycle_B", "idt_B", "G_B_black_consis"
        ]

        # Define visualizations for domain A and domain B
        visual_names_A = ["real_A", "fake_B_full", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "real_matte", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")
        self.visual_names = visual_names_A + visual_names_B

        # Define model names for saving/loading
        self.model_names = ["G_A", "G_B", "D_A", "D_B"] if self.isTrain else ["G_A", "G_B"]

        # Define networks (generators and discriminators)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # Initialize image pools for fake images
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # Define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()

            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary preprocessing.

        Args:
            input (dict): Contains the data and metadata information.
        """
        self.real_matte = input["matte"].to(self.device)
        self.real_A = input["A"].to(self.device)
        self.real_A_free = input["A_free"].to(self.device)
        self.real_B = input["B"].to(self.device)
        self.image_paths = input["AB_paths"]

    def forward(self):
        """Run forward pass to compute fake images and reconstruction images."""
        self.fake_B = self.netG_A(self.real_A)  # Generate shadow image
        self.real_matte_mask = (self.real_matte + 1) / 2  # Convert matte tensor to mask
        self.fake_B_full = self.fake_B * self.real_matte_mask + self.real_A_free * (1 - self.real_matte_mask)
        self.rec_A = self.netG_B(self.fake_B)  # Reconstruct shadow-free image
        self.fake_A = self.netG_B(self.real_B)  # Generate shadow-free image
        self.rec_B = self.netG_A(self.fake_A)  # Reconstruct shadow image

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator.

        Args:
            netD: Discriminator network.
            real: Real images.
            fake: Fake images generated by the generator.

        Returns:
            Tensor: Discriminator loss.
        """
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A."""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B."""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B."""
        lambda_idt = self.opt.lambda_identity
        lambda_L1 = self.opt.lambda_L1
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN losses
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Shadow consistency losses
        self.loss_G_A_black_consis = self.criterionL1(self.real_A * (1 - self.real_matte), self.fake_B * (1 - self.real_matte)) * lambda_L1
        self.loss_G_B_black_consis = self.criterionL1(self.real_B * (1 - self.real_matte), self.fake_A * (1 - self.real_matte)) * lambda_L1

        # Cycle consistency losses
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_A_black_consis + self.loss_G_B_black_consis
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights."""
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()