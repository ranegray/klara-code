from glob import glob

from setuptools import find_packages, setup

package_name = 'xlerobot_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rane Gray',
    maintainer_email='ranegray830@gmail.com',
    description="Driver for the XLeRobot's Feetech STS3215 servo buses.",
    license='MIT',
    entry_points={
        'console_scripts': [
            'servo_bus = xlerobot_driver.cli:main',
        ],
    },
)
