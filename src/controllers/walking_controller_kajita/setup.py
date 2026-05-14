from setuptools import find_packages, setup

package_name = 'walking_controller_kajita'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eliascm',
    maintainer_email='jose.cabeza@utec.edu.pe',
    description='Walking Controller Kajita: Controller based on LIPM with ZMP Preview Control',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'my_node = my_py_pkg.my_node:main',
            
        ],
    },
)
