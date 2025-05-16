import os
import subprocess
import sys

def create_virtual_env(venv_name=".venv"):
    print(f"🔧 Creating virtual environment: {venv_name}")
    subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)

    # Determine correct pip path
    if os.name == 'nt':
        pip_path = os.path.join(venv_name, "Scripts", "pip.exe")
        activate_cmd = f"{venv_name}\\Scripts\\activate"
    else:
        pip_path = os.path.join(venv_name, "bin", "pip")
        activate_cmd = f"source {venv_name}/bin/activate"

    print("📦 Installing dependencies from requirements.txt...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)

    print(f"\n✅ Setup complete!")
    print(f"👉 To activate your environment, run:\n\n    {activate_cmd}\n")

if __name__ == "__main__":
    create_virtual_env()
