from setuptools import find_packages, setup

setup(
    name="lerobot_sim2real",
    version="0.0.2",
    description="Sim2Real Manipulation with LeRobot SO101",
    url="https://github.com/Luoyadan/lerobot_so101-sim2real",
    packages=find_packages(include=["lerobot_sim2real*"]),
    python_requires=">=3.9",
    setup_requires=["setuptools>=62.3.0"],
    install_requires=[
        "tensorboard",
        "wandb"
    ]
)
