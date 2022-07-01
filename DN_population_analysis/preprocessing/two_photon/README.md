# DN_population_analysis

The code should be executed in the following order:
1. convert_from_raw.py
2. aligne_and_denoise.py (pip install git+https://github.com/NeLy-EPFL/noisy2way@v0.0.1)
3. motion_correction.py (https://github.com/NeLy-EPFL/ofco@v0.0.1)
4. warp_green.py
(pip install git+https://github.com/NeLy-EPFL/deepinterpolation@8c86b279f01e8ec5e3f1a85233ae1ac76445fb3e)
5. denoise_green.py
6. dff_stack.py
