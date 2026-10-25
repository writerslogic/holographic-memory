#!/usr/bin/env python3
"""Export a transparent APNG and a separate white/black comparison preview.

The artwork stays transparent. Only the comparison GIF uses test backgrounds.
Requires Pillow, NumPy and ffmpeg. Uses the existing 25 fps RGBA master.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--apng', type=Path, required=True)
    parser.add_argument('--preview', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(p for p in args.frames.iterdir()
                   if re.fullmatch(r'frame-\d{4}\.png', p.name))
    if len(files) != 250:
        raise RuntimeError(f'Expected 250 complete master frames, found {len(files)}')
    for i, file in enumerate(files):
        if file.name != f'frame-{i:04d}.png':
            raise RuntimeError('The frame sequence is not contiguous.')
    args.apng.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    apng_temporary = args.apng.with_name(args.apng.stem + '.writing.apng')
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-framerate', '25', '-i', str(args.frames/'frame-%04d.png'),
        '-frames:v', '250', '-c:v', 'apng', '-pix_fmt', 'rgba',
        '-plays', '0', '-f', 'apng', str(apng_temporary),
    ], check=True)
    report = {}
    with Image.open(apng_temporary) as animation:
        if animation.n_frames != 250 or animation.info.get('loop') != 0:
            raise RuntimeError('APNG frame count or loop metadata is incorrect.')
        duration = 0
        for i in range(animation.n_frames):
            animation.seek(i)
            animation.load()
            duration += animation.info.get('duration', 0)
            if i in (0, 62, 125, 188, 249):
                actual = np.asarray(animation.convert('RGBA'))
                with Image.open(files[i]) as master:
                    expected = np.asarray(master.convert('RGBA'))
                if not np.array_equal(actual[:,:,3], expected[:,:,3]):
                    raise RuntimeError(f'APNG alpha mismatch at frame {i}')
                visible = expected[:,:,3] > 0
                if not np.array_equal(actual[visible], expected[visible]):
                    raise RuntimeError(f'APNG visible-pixel mismatch at frame {i}')
        if abs(duration - 10000) > 0.001:
            raise RuntimeError(f'Unexpected APNG duration: {duration}')
        report['apng'] = {'frames':250, 'duration_ms':duration,
                          'loop':0, 'transparent':True, 'matches_master':True}
    apng_temporary.replace(args.apng)
    print('Transparent APNG encoded and verified.', flush=True)

    size = 512
    white = Image.new('RGBA', (size,size), '#ffffff')
    black = Image.new('RGBA', (size,size), '#000000')
    palette = [c for value in range(256) for c in (value,value,value)]
    previews = []
    for file in files:
        with Image.open(file) as master:
            mark = master.convert('RGBA').resize((size,size), Image.Resampling.LANCZOS)
        panel = Image.new('RGB', (size*2,size))
        panel.paste(Image.alpha_composite(white,mark).convert('RGB'), (0,0))
        panel.paste(Image.alpha_composite(black,mark).convert('RGB'), (size,0))
        frame = panel.convert('L').convert('P')
        frame.putpalette(palette)
        previews.append(frame)
    preview_temporary = args.preview.with_name(args.preview.stem+'.writing.gif')
    previews[0].save(preview_temporary, save_all=True, append_images=previews[1:],
                     duration=40, loop=0, optimize=True, disposal=1)
    del previews
    with Image.open(preview_temporary) as preview:
        if preview.n_frames != 250 or preview.info.get('loop') != 0:
            raise RuntimeError('Preview does not contain the complete loop.')
        duration = 0
        for i in range(preview.n_frames):
            preview.seek(i)
            preview.load()
            duration += preview.info.get('duration',0)
        if duration != 10000:
            raise RuntimeError('Unexpected preview duration.')
        report['preview'] = {'frames':250, 'duration_ms':duration, 'loop':0,
                             'size':[1024,512], 'test_backgrounds':['white','black']}
    preview_temporary.replace(args.preview)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
