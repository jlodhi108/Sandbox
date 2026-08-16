import os
from github import Github, InputGitAuthor


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
