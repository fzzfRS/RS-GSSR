from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument(
            "--ntest", type=int, default=float("inf"), help="# of test examples."
        )
        parser.add_argument(
            "--GSS_results_dir",
            type=str,
            default="./datasets/RS_SynShadow",
            help="saves results here.",
        )
        parser.add_argument(
            "--GSR_results_dir",
            type=str,
            default="./results",
            help="saves results here.",
        )
        parser.add_argument(
            "--aspect_ratio",
            type=float,
            default=1.0,
            help="aspect ratio of result images",
        )
        parser.add_argument(
            "--phase", type=str, default="test", help="train, val, test, etc"
        )

        parser.add_argument(
            "--eval", type=bool, default=True, help="use eval mode during test time."
        )
        parser.add_argument(
            "--num_test", type=int, default=10000, help="how many test images to run"
        )

        # # rewrite devalue values
        parser.set_defaults(load_size=parser.get_default("crop_size"))
        self.isTrain = False
        return parser
