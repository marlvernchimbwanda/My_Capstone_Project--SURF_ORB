from setuptools import find_packages, setup

package_name = 'monocular_visual_odom'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='uozrobotics',
    maintainer_email='marlvernchimbwanda1002@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "surf_detector = monocular_visual_odom.surf_detector:main" # package name -> file name : Function to execute
        ],
    },
)
