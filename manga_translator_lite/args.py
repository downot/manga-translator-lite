"""Argparse for the three-step pipeline.

Subcommands:

    extract   detect + OCR + inpaint → workspace
    translate hit the LLM, fill in pages.json
    render    paint translations onto clean images
    run       extract + translate + render in one go
    config-help  print the config schema
"""
import argparse
import os
from urllib.parse import unquote


def _url_decode(s: str) -> str:
    s = unquote(s)
    if s.startswith('file:///'):
        s = s[len('file://'):]
    return s


def path_type(string: str) -> str:
    if not string:
        return ''
    s = _url_decode(os.path.expanduser(string))
    if not os.path.exists(s):
        raise argparse.ArgumentTypeError(f'No such file or directory: "{string}"')
    return s


def file_path_type(string: str) -> str:
    if not string:
        return ''
    s = _url_decode(os.path.expanduser(string))
    if not os.path.isfile(s):
        raise argparse.ArgumentTypeError(f'No such file: "{string}"')
    return s


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument('-c', '--config', default=None, type=file_path_type,
                   help='Path to the .toml or .json pipeline config file.')
    p.add_argument('--target-lang', default=None,
                   help='Override translator.target_lang (CHS, ENG, JPN, ...).')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Verbose logging and intermediate diagnostics.')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='manga_translator_lite',
        description='Local OCR + third-party LLM manga translation pipeline.',
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_extract = sub.add_parser('extract',
                               help='Step 1: run detection / OCR / inpaint and write the workspace.')
    p_extract.add_argument('-i', '--input', required=True, type=path_type,
                           help='Input image file or directory of images.')
    p_extract.add_argument('-w', '--work-dir', required=True,
                           help='Workspace directory to create / update.')
    p_extract.add_argument('--overwrite', action='store_true',
                           help='Re-extract all images even if they already exist in the workspace.')
    _add_common(p_extract)

    p_tx = sub.add_parser('translate',
                          help='Step 2: call the LLM, fill in translations.')
    p_tx.add_argument('work_dir', help='Existing workspace directory.')
    p_tx.add_argument('--overwrite', action='store_true',
                      help='Re-translate even blocks that already have translations.')
    p_tx.add_argument('--start-index', type=int, default=None,
                      help='Starting page index in pages.json to (re)translate from.')
    _add_common(p_tx)

    p_render = sub.add_parser('render',
                              help='Step 3: render translations onto clean images.')
    p_render.add_argument('work_dir', help='Existing workspace directory.')
    p_render.add_argument('-o', '--output', required=True,
                          help='Output directory for final images.')
    _add_common(p_render)

    p_run = sub.add_parser('run',
                           help='Run extract + translate + render end-to-end.')
    p_run.add_argument('-i', '--input', required=True, type=path_type,
                       help='Input image file or directory of images.')
    p_run.add_argument('-w', '--work-dir', required=True,
                       help='Workspace directory to create / update.')
    p_run.add_argument('-o', '--output', required=True,
                       help='Output directory for final images.')
    p_run.add_argument('--overwrite', action='store_true',
                       help='Re-extract and re-translate even if results already exist.')
    _add_common(p_run)

    sub.add_parser('config-help',
                   help='Print the JSON schema of the config file.')

    return parser
