"""Print a real streamed response and optionally inspect its first chunks."""

import argparse
import sys

from pydantic import ValidationError

from resolv.chat import stream_chat
from resolv.core.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="show the first empty and 3 text chunks"
    )
    args = parser.parse_args()

    try:
        Settings()
    except ValidationError as exc:
        raise SystemExit(
            "Check OPENAI_API_KEY and OPENAI_BASE_URL in .env or the environment."
        ) from exc

    question = "请简单介绍一下 Resolv 能帮助用户做什么。"
    text_chunks_shown = 0
    empty_chunk_shown = False
    for index, chunk in enumerate(stream_chat(question)):
        show_empty = not chunk.text and not empty_chunk_shown
        show_text = bool(chunk.text) and text_chunks_shown < 3
        if args.debug and (show_empty or show_text):
            print(
                f"chunk {index}: {type(chunk).__name__}, text={chunk.text!r}",
                file=sys.stderr,
            )
            if show_empty:
                empty_chunk_shown = True
            else:
                text_chunks_shown += 1
        print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
