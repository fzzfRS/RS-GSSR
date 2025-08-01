from . import networks
from .base_model import BaseModel


class TestGSSModel(BaseModel):
    """This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, "TestModel cannot be used during training time"
        # parser.add_argument('--model_suffix', type=str, default='',
        #                     help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.add_argument(
            "--model_suffix",
            type=str,
            default="_A",
            help="In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.",
        )

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert not opt.isTrain
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ["shadow", "mask", "shadowfree"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ["G" + opt.model_suffix]  # only generator is needed.
        # self.model_names = ['G']
        self.netG_A = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """

        self.mask = input["matte"].to(self.device)
        self.real = input["A"].to(self.device)
        self.shadowfree = input["A_free"].to(self.device)
        self.image_paths = input["AB_paths"]

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG_A(self.real)  # G_A(A)

        # Transfer matte_tensor to range [0, 1]
        self.real_matte_mask = (
            self.mask + 1
        ) / 2  # Assuming real_matte is in the range [0, 1]
        self.shadow = self.fake * self.real_matte_mask + self.shadowfree * (
            1 - self.real_matte_mask
        )
        # Limit tensor to range [-1, 1]

        self.shadow = torch.clamp(self.shadow, min=-1.0, max=1.0)

    def optimize_parameters(self):
        """No optimization for test model."""
