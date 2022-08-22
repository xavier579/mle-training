import argparse

_DEFAULT = argparse.Namespace(in_dir=None, out_dir=None)
args = _DEFAULT


def command_line_args():

    parser = argparse.ArgumentParser(prog="train", description="Does magic :) .")
    parser.add_argument("--data_folder", default="../../data", help="Path for input data")
    parser.add_argument("--output_folder", default="../../artifacts", help="Path for model output")

    parser.add_argument("--log_level", default="DEBUG", help="Log level")
    parser.add_argument(
        "--log_path",
        default=None,
        help="Path to store logs, if empty logs would not be written to a file",
    )
    # parser.add_argument('--no_console_log', default='true', help='true if logs to be written to console, else false')
    parser.add_argument(
        "--no_console_log",
        action=argparse.BooleanOptionalAction,
        help="true if logs to be written to console, else false",
    )
    return parser


def set_args(cmd_line=None):

    parser = command_line_args()
    global args
    args = parser.parse_args(cmd_line)
