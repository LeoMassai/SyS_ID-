import os
import sys
import subprocess

ENV_NAME = ".venv" if os.name != "nt" else "venv"


def create_virtualenv():
    """Creates a virtual environment if it does not exist."""
    if not os.path.exists(ENV_NAME):
        print(f"Creating virtual environment '{ENV_NAME}'...")
        subprocess.run([sys.executable, "-m", "venv", ENV_NAME], check=True)
    else:
        print(f"The virtual environment '{ENV_NAME}' already exists.")


def install_dependencies():
    """Installs dependencies from requirements.txt."""
    python_executable = os.path.join(ENV_NAME, "bin", "python") if os.name != "nt" else os.path.join(ENV_NAME,
                                                                                                     "Scripts",
                                                                                                     "python.exe")
    print("Upgrading pip...")
    subprocess.run([python_executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print("Installing dependencies from requirements.txt...")
    subprocess.run([python_executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)


def main():
    create_virtualenv()
    install_dependencies()
    print("\nInstallation complete. Please activate the virtual environment in PyCharm.")
    if os.name != "nt":
        print("To activate the virtual environment, run:\n    source .venv/bin/activate")
    else:
        print("To activate the virtual environment, run:\n    .\\venv\\Scripts\\activate")


if __name__ == "__main__":
    main()
