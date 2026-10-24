"""Run the local MROOT web interface with: python app.py."""

from src.mroot.ui import create_demo


if __name__ == "__main__":
    create_demo().queue(default_concurrency_limit=1, max_size=8).launch(max_file_size="20mb")
