import os
from github import Github, InputGitAuthor, InputGitTreeElement


def open_modernization_pr(
    file_path: str,
    new_content: str,
    branch_name: str,
    pr_title: str,
    pr_body: str,
) -> str:
    token = os.environ["GITHUB_TOKEN"]
    repo_name = os.environ["GITHUB_REPO"]

    gh = Github(token)
    repo = gh.get_repo(repo_name)

    base_branch = repo.default_branch
    base_ref = repo.get_git_ref(f"heads/{base_branch}")
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_ref.object.sha)

    contents = repo.get_contents(file_path, ref=base_branch)
    repo.update_file(
        path=file_path,
        message=f"chore: modernize {file_path}",
        content=new_content,
        sha=contents.sha,
        branch=branch_name,
    )

    pr = repo.create_pull(
        title=pr_title,
        body=pr_body,
        head=branch_name,
        base=base_branch,
    )
    return pr.html_url


def open_multi_file_pr(
    files: list[tuple[str, str]],  # [(file_path, new_content), ...]
    branch_name: str,
    pr_title: str,
    pr_body: str,
) -> str:
    """Commit multiple files in ONE commit on a new branch and open ONE
    PR. Unlike open_modernization_pr (single file, uses the simple
    update_file() convenience call), GitHub has no single-call way to
    update several files atomically — this uses PyGithub's lower-level
    git data API instead: create a blob per file, build a tree from the
    base tree plus those blobs, commit that tree, then point the branch
    at the new commit."""
    token = os.environ["GITHUB_TOKEN"]
    repo_name = os.environ["GITHUB_REPO"]

    gh = Github(token)
    repo = gh.get_repo(repo_name)

    base_branch = repo.default_branch
    base_ref = repo.get_git_ref(f"heads/{base_branch}")
    base_commit = repo.get_git_commit(base_ref.object.sha)

    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_ref.object.sha)

    tree_elements = [
        InputGitTreeElement(path=file_path, mode="100644", type="blob", content=new_content)
        for file_path, new_content in files
    ]
    new_tree = repo.create_git_tree(tree_elements, base_commit.tree)
    new_commit = repo.create_git_commit(
        message=f"chore: modernize {len(files)} file(s)",
        tree=new_tree,
        parents=[base_commit],
    )

    branch_ref = repo.get_git_ref(f"heads/{branch_name}")
    branch_ref.edit(sha=new_commit.sha)

    pr = repo.create_pull(
        title=pr_title,
        body=pr_body,
        head=branch_name,
        base=base_branch,
    )
    return pr.html_url
