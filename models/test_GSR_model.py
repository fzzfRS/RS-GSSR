# Python
from . import networks
from .base_model import BaseModel


class TestGSRModel(BaseModel):
    """
    This class implements the test model for Generative Shadow Removal (GSR).

    The model is used for testing shadow removal tasks in one direction only.
    It automatically sets '--dataset_mode HEI_single', which loads images from a single domain.
    The model requires the '--model_suffix' option to specify the generator network.

    Note:
        This model is only used during testing and cannot be used for training.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options and rewrite default values for existing options.

        Args:
            parser: Original option parser.
            is_train (bool): Whether training phase or test phase.

        Returns:
            argparse.ArgumentParser: The modified parser with additional options.

        Raises:
            AssertionError: If the model is used during training.
        """
        assert not is_train, "TestGSRModel cannot be used during training time."
        parser.set_defaults(dataset_mode="HEI_single")
        parser.add_argument(
            "--model_suffix",
            type=str,
            default="",
            help="In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.",
        )
        return parser

    def __init__(self, opt):
        """Initialize the test model.

        Args:
            opt: Options containing model configurations.

        Raises:
            AssertionError: If the model is used during training.
        """
        assert not opt.isTrain
        super().__init__(opt)

        # Define loss names (empty for test model)
        self.loss_names = []

        # Define visualizations for testing
        self.visual_names = ["fake"]

        # Define model names for saving/loading
        self.model_names = ["G"]

        # Initialize the generator network
        self.netG = networks.define_G(
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
        """Unpack input data from the dataloader and perform necessary preprocessing.

        Args:
            input (dict): Contains the data and metadata information.
        """
        self.shadow = input["umbra"].to(self.device)
        self.HEI = input["de"].to(self.device)
        self.mask = input["umbra_mask"].to(self.device)
        self.edge = input["edge_mask"].to(self.device)
        self.image_paths = input["A_paths"]

    def forward(self):
        """Run forward pass to generate shadow-free images."""
        self.fake = self.netG(self.shadow, self.HEI, self.mask, self.edge)

    def optimize_parameters(self):
        """No optimization is performed for the test model."""
        pass