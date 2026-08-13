import subprocess

with open("pyproject.toml", "r") as f:
    content = f.read()

content = content.replace("python_version = \"3.10\"", "python_version = \"3.12\"")

with open("pyproject.toml", "w") as f:
    f.write(content)

subprocess.run(["mypy", "--strict", "snapvec/"])

with open("pyproject.toml", "r") as f:
    content = f.read()

content = content.replace("python_version = \"3.12\"", "python_version = \"3.10\"")

with open("pyproject.toml", "w") as f:
    f.write(content)
