import sys

from sources.crowd_recital.download import main as crowd_recital_main
from sources.knesset.download import main as knesset_main

source_to_main = {
    "crowd_recital": crowd_recital_main,
    "knesset": knesset_main,
}


def main():
    # First arg after script name should be the source
    if len(sys.argv) < 2:
        print("Error: Source must be specified")
        print("Usage: python download.py <source> [source-specific-args]")
        print(f"Supported sources: {', '.join(source_to_main.keys())}")
        sys.exit(1)

    source = sys.argv[1]
    if source in source_to_main:
        main = source_to_main[source]
        # Remove the script name and source argument, pass the rest to the module
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main()
    else:
        print(f"Error: Source '{source}' is not supported.")
        print(f"Supported sources: {', '.join(source_to_main.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
