import argparse
import os
from dotenv import load_dotenv

from languages import get_handler
from agents.graph import modernize
from git_ops.pr import open_modernization_pr

load_dotenv()


def run(file_path: str, open_pr: bool, max_iterations: int) -> None:
    handler = get_handler(file_path)
    with open(file_path, "rb") as f:
        source = f.read()

    chunks = handler.chunk(source)
    print(f"[{handler.name}] Found {len(chunks)} chunk(s) to modernize in {file_path}")

    # Process bottom-to-top and accumulate into `working_source` so each
    # chunk is verified against everything already modernized before it
    # (not just the pristine original). Editing from the end of the file
    # backward keeps every not-yet-processed chunk's byte offsets valid,
    # since a splice never shifts bytes that come before it.
    working_source = source
    succeeded = 0
    collected_imports: list[str] = []  # merged in once, AFTER the loop —
    # prepending mid-loop would shift byte offsets for chunks not yet
    # processed, since they all sit before the insertion point
    for chunk in sorted(chunks, key=lambda c: c.start_byte, reverse=True):
        print(f"\n--- Modernizing {chunk.kind} [{chunk.start_byte}:{chunk.end_byte}] ---")
        final_state = modernize(
            handler.name, working_source, chunk.start_byte, chunk.end_byte,
            max_iterations=max_iterations,
        )
        print(f"status={final_state['status']} iterations={final_state['iteration_count']}")

        if final_state["status"] == "success":
            new_code = final_state["modernized_code"].encode("utf-8")
            working_source = (
                working_source[:chunk.start_byte]
                + new_code
                + working_source[chunk.end_byte:]
            )
            for m in final_state.get("required_imports", []):
                if m not in collected_imports:
                    collected_imports.append(m)
            succeeded += 1
        else:
            print(f"SKIPPED (gave up): {chunk.kind} at byte {chunk.start_byte}")
            print(f"last error:\n{final_state['compiler_stderr']}")

    if succeeded == 0:
        print("\nNo chunks were successfully modernized. Exiting.")
        return

    existing_text = working_source.decode("utf-8")
    missing_imports = [m for m in collected_imports if not handler.has_import(existing_text, m)]
    if missing_imports:
        header_block = "".join(handler.import_statement(m) for m in missing_imports)
        working_source = header_block.encode("utf-8") + working_source

    new_source = working_source
    root, ext = os.path.splitext(file_path)
    output_path = f"{root}.modernized{ext}"
    with open(output_path, "wb") as f:
        f.write(new_source)
    print(f"\nWrote modernized file to {output_path} ({succeeded}/{len(chunks)} chunks modernized)")

    if open_pr:
        url = open_modernization_pr(
            file_path=file_path,
            new_content=new_source.decode("utf-8"),
            branch_name=f"chore/modernize-{file_path.replace('/', '-')}",
            pr_title=f"chore: modernize {file_path}",
            pr_body=(
                f"Automated modernization via code-modernizer.\n\n"
                f"{succeeded}/{len(chunks)} chunks successfully modernized "
                f"and verified in an isolated sandbox."
            ),
        )
        print(f"Opened PR: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Code Modernization Engine")
    parser.add_argument("file", help="Path to the legacy source file to modernize")
    parser.add_argument("--pr", action="store_true", help="Open a GitHub PR with the result")
    parser.add_argument("--max-iterations", type=int, default=5)
    args = parser.parse_args()

    run(args.file, args.pr, args.max_iterations)
