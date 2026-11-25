# flower-ascii-art

A 3D ASCII flower that blooms, then blows away. Made with Python, numpy and pygame.

[Watch the video](bloom/bloom3d.mp4)

## Run

    git clone https://github.com/edwardjones7/flower-ascii-art.git
    cd flower-ascii-art
    pip install -r requirements.txt
    python bloom/bloom3d.py

Space restarts, Esc quits. Uses the Consolas font (included with Windows).
Other systems may fall back to another font, so the look can differ.

## Export a video (needs ffmpeg)

    python bloom/bloom3d.py --record
    ffmpeg -framerate 30 -i frames3d/f_%04d.png -c:v libx264 -pix_fmt yuv420p bloom3d.mp4

By @edjonesai
