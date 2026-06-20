import asyncio
import json
import logging
import sys

from .args import build_parser
from .config import Config
from .utils import get_logger, init_logging, set_log_level


def _resolve_reference_langs(args, cfg):
    """Combine CLI flags and config into the reference-langs trichotomy:
    None = auto, [] = off, [codes...] = manual. CLI overrides config."""
    if getattr(args, 'no_reference', False):
        return []                                  # CLI: explicit off
    cli = getattr(args, 'reference_lang', None)
    if cli:
        return cli                                 # CLI: explicit manual list
    return cfg.translator.reference_langs          # fall back to config (default None=auto)


async def _dispatch(args) -> int:
    cfg = Config.load(args.config)
    if args.target_lang:
        cfg.translator.target_lang = args.target_lang

    if args.cmd == 'extract':
        from .pipeline.extract import run_extract
        await run_extract(args.input, args.work_dir, cfg, verbose=args.verbose,
                          target_lang=args.target_lang, overwrite=args.overwrite)
        return 0

    if args.cmd == 'translate':
        from .pipeline.translate import run_translate
        await run_translate(args.work_dir, cfg, overwrite=args.overwrite,
                            target_lang=args.target_lang, start_index=args.start_index,
                            reference_langs=_resolve_reference_langs(args, cfg))
        return 0

    if args.cmd == 'render':
        from .pipeline.render import run_render
        await run_render(args.work_dir, args.output, cfg,
                         check=getattr(args, 'check', False),
                         no_check=getattr(args, 'no_check', False),
                         yes=getattr(args, 'yes', False))
        return 0

    if args.cmd == 'run':
        from .pipeline.extract import run_extract
        from .pipeline.render import run_render
        from .pipeline.translate import run_translate
        await run_extract(args.input, args.work_dir, cfg, verbose=args.verbose,
                          target_lang=args.target_lang, overwrite=args.overwrite)
        await run_translate(args.work_dir, cfg, target_lang=args.target_lang,
                            reference_langs=_resolve_reference_langs(args, cfg))
        await run_render(args.work_dir, args.output, cfg,
                         check=getattr(args, 'check', False),
                         no_check=getattr(args, 'no_check', False),
                         yes=getattr(args, 'yes', False))
        return 0

    if args.cmd == 'config-help':
        print(json.dumps(Config.model_json_schema(), indent=2))
        return 0

    return 1


def main() -> int:
    init_logging()
    parser = build_parser()
    args = parser.parse_args()
    set_log_level(level=logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO)
    logger = get_logger(args.cmd)
    try:
        return asyncio.run(_dispatch(args))
    except KeyboardInterrupt:
        print('\nCancelled by user.', file=sys.stderr)
        return 130
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if getattr(args, 'verbose', False) else None)
        return 1


if __name__ == '__main__':
    sys.exit(main())
