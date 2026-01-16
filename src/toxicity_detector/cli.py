"""Command-line interface for the Toxicity Detector."""

import argparse
import sys
from loguru import logger

from toxicity_detector import detect_toxicity, PipelineConfig
from toxicity_detector.datamodels import ToxicityType


def detect_command(args):
    """Run toxicity detection on input text."""
    # Load pipeline configuration
    try:
        pipeline_config = PipelineConfig.from_file(args.pipeline_config)
    except Exception as e:
        logger.error(
            f"Failed to load pipeline config from '{args.pipeline_config}': {e}"
        )
        sys.exit(1)

    # Validate toxicity type
    if args.toxicity_type not in ToxicityType._value2member_map_:
        logger.error(
            f"Invalid toxicity type '{args.toxicity_type}'. "
            f"Valid options: {list(ToxicityType._value2member_map_.keys())}"
        )
        sys.exit(1)

    # Run toxicity detection
    try:
        result = detect_toxicity(
            input_text=args.text,
            user_input_source=args.source,
            toxicity_type=args.toxicity_type,
            context_info=args.context,
            pipeline_config=pipeline_config,
            serialize_result=args.save,
        )

        # Display results
        print("\n" + "=" * 80)
        print("TOXICITY DETECTION RESULT")
        print("=" * 80)
        print(f"\nInput text: {args.text}")
        print(f"Toxicity type: {args.toxicity_type}")
        print(f"\nContains toxicity: {result.answer.get('contains_toxicity', 'N/A')}")

        if args.verbose:
            print(f"\nFull analysis: {result.answer.get('analysis_result', 'N/A')}")

        if args.save:
            print(f"\nResult saved to: {pipeline_config.result_data_path}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Error during toxicity detection: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def app_command(args):
    """Start the Gradio app."""
    # Import here to avoid loading Gradio unless needed
    from toxicity_detector.app import launch_app

    config_path = args.app_config or args.pipeline_config
    config_type = "app" if args.app_config else "pipeline"
    logger.info(f"Starting app with {config_type} config: {config_path}")

    try:
        launch_app(
            config_path=config_path,
            config_type=config_type,
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
        )
    except Exception as e:
        logger.error(f"Failed to start app: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="toxicity-detector",
        description="LLM-based pipeline to detect toxic speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run toxicity detection on text
  toxicity-detector detect --text "Your text here" --pipeline-config config/pipeline_config.yaml

  # Start the Gradio app with app config
  toxicity-detector app --app-config config/app_config.yaml

  # Start the Gradio app with pipeline config (uses default app settings)
  toxicity-detector app --pipeline-config config/pipeline_config.yaml --share
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Detect subcommand
    detect_parser = subparsers.add_parser(
        "detect",
        help="Run toxicity detection on input text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    detect_parser.add_argument(
        "--text",
        "-t",
        required=True,
        help="Text to analyze for toxicity",
    )
    detect_parser.add_argument(
        "--pipeline-config",
        "-p",
        required=True,
        help="Path to pipeline configuration YAML file (required)",
    )
    detect_parser.add_argument(
        "--toxicity-type",
        "-T",
        default="hatespeech",
        help=(
            "Type of toxicity to detect 'personlized_toxicity' or "
            "'hatespeech' (default: 'hatespeech')"
        ),
    )
    detect_parser.add_argument(
        "--source",
        "-s",
        default=None,
        help="Source identifier for the input (e.g., 'chat', 'comment')",
    )
    detect_parser.add_argument(
        "--context",
        "-c",
        default=None,
        help="Additional context about the conversation or situation",
    )
    detect_parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save the result to disk as YAML",
    )
    detect_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    detect_parser.set_defaults(func=detect_command)

    # App subcommand
    app_parser = subparsers.add_parser(
        "app",
        help="Start the Gradio web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    config_group = app_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--app-config",
        "-a",
        help="Path to app configuration YAML file",
    )
    config_group.add_argument(
        "--pipeline-config",
        "-p",
        help="Path to pipeline configuration YAML file (uses default app settings)",
    )

    app_parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Server name (default: 127.0.0.1)",
    )
    app_parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    app_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    app_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    app_parser.set_defaults(func=app_command)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the appropriate command
    args.func(args)


if __name__ == "__main__":
    main()
