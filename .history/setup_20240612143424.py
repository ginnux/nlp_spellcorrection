from cx_Freeze import setup, Executable

options = {
    "build_exe": {
        "packages": ["openpyxl"],
    }
}

setup(
    name="NLP",
    version="1.0",
    description="Your Description",
    options=options,
    executables=[Executable("UI.py")],
)
