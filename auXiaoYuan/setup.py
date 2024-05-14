from setuptools import setup, find_packages

requirements = []
with open("./requirements.txt", "r") as f:
    requirements = [r.strip("\n").strip(" ") for r in f.readlines()]


setup(
    name='aYuan',
    version='0.1',
    author='auXiaoYuan',
    author_email='heyuanYin@qq.com',
    description='a package with some useful module for computer vision',
    packages=find_packages(),
    install_requires=requirements,
)