This repository contains the code for the explanatory videos on the [sigmanought YouTube channel](https://www.youtube.com/channel/UCN0AySD8Xxad8H2qsXEZ1wg).

Most of the code is generated with [Manim Community](https://github.com/ManimCommunity/manim/). The majority of SAR and optical images used in the videos are subject to licensing restrictions, so placeholder images are included instead. You can usually find the original sources of the images through the attributions shown in the videos. I’m sharing the code to inspire others and show the (more practical than perfect) logic behind the animations. Therefore, the logo and custom backgrounds are not included.

All contents are available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). This means you can use the videos for non-commercial purposes under the same license, just remember to give credit to sigmanought.


# 01 - Why does SAR use different wavelengths? | SAR bands explained
In this video we'll explore why SAR uses different wavelengths.

We'll look at the applications for different bands and on which objects the bands reflect particularly well. Of course, we'll be looking at a beautiful real-world SAR image for each group of bands!

In the end, we'll also find out the (alleged) history behind the names of the SAR bands. Whether fact or urban legend, the stories are useful for remembering the cryptic bands names.

# Setup
To run a manim file:
1. Activate the environment
    Install uv: ``curl -LsSf https://astral.sh/uv/install.sh | sh`` \
    Install dependencies: ``uv sync`` \
    Install repository as package ``uv pip install -e .`` \
    Activate: ``source .venv/bin/activate`` \
    You may also need to install [cairo](https://pycairo.readthedocs.io/en/latest/getting_started.html) and [manim](https://docs.manim.community/en/stable/installation.html). 
2. Run the desired class ``manim -p -ql example.py SquareToCircle``

Scripts that contain *opengl* in their name, are rendered via
``manim ./example.py --renderer=opengl --format=mp4``.
To render with higher resolution use ``-qh`` (high quality) or ``-qk`` (4k).