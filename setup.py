from cx_Freeze import setup, Executable

options = {
    "build_exe": {
        "packages": ["nltk"],
    }
}

setup(
    name="NLP",
    version="1.0",
    description="Your Description",
    options=options,
    executables=[Executable("UI.py")],
    include_files=[
        ("vocab.txt", "./vocab.txt"),
        ("count_1edit.txt", "./count_1edit.txt"),
    ],
)
