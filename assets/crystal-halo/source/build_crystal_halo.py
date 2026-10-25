#!/usr/bin/env python3
"""Transparent ink crystal with a crisp white halo and seamless 3D rotation.

Dependencies: numpy, scipy, Pillow with WebP support, matplotlib.
Run --samples for design previews, or --full for production exports.
All final raster animations use the same 250-frame RGBA master at 25 fps.
Add --resume only to continue an interrupted export with unchanged render settings.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exports"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "mpl-cache"))
import numpy as np
from PIL import Image, ImageDraw, _webp
from scipy.spatial import ConvexHull
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

SIZE = 768
DURATION = 10
FPS = 25
INK = (29, 31, 30)
HALO_SCALE = 1.0
# In lossless mode this controls compression effort, not pixel fidelity.
WEBP_EFFORT = 75
WEBP_METHOD = 3


def save_image_atomic(image, path, **options):
    path = Path(path)
    temporary = path.with_name(path.stem + ".writing" + path.suffix)
    image_format = {".jpg":"JPEG", ".png":"PNG", ".gif":"GIF", ".webp":"WEBP"}[path.suffix]
    with open(temporary, "wb") as handle:
        image.save(handle, format=image_format, **options)
        handle.flush()
        os.fsync(handle.fileno())
    if temporary.stat().st_size == 0:
        raise RuntimeError(f"Empty image output: {temporary}")
    temporary.replace(path)


def write_atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".writing")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def make_geometry():
    # Broad irregular crown, side shoulders and an elongated converging base.
    outer = np.array([
        [.58, 1.90, .10], [-.47, 1.42, .25], [-1.40, .20, .02],
        [1.17, .70, .10], [-.20, -1.85, .00], [.08, .75, 1.02],
        [-.38, -.60, .82], [.70, -.56, .36], [-.20, 1.12, -.97],
        [-1.06, .00, -.79], [.35, -.67, -.88], [1.05, .60, -.59],
    ], dtype=float)
    blends = [
        [(5,.55),(0,.20),(1,.25)], [(2,.60),(5,.25),(6,.15)],
        [(5,.42),(6,.33),(3,.25)], [(6,.45),(2,.25),(4,.30)],
        [(7,.50),(5,.20),(4,.30)], [(8,.40),(11,.25),(0,.35)],
        [(9,.40),(8,.30),(6,.30)], [(10,.45),(11,.30),(5,.25)],
    ]
    inside = np.array([sum(outer[i]*w for i,w in blend) for blend in blends])
    vertices = np.concatenate([outer, inside])
    hull = ConvexHull(outer)
    edge_faces = {}
    for face_index, face in enumerate(hull.simplices):
        for i in range(3):
            pair = tuple(sorted((int(face[i]),int(face[(i+1)%3]))))
            edge_faces.setdefault(pair, []).append(face_index)
    edges = [{"vertices":list(pair),"faces":faces,"kind":"scaffold"}
             for pair,faces in sorted(edge_faces.items())]
    links = [
        (12,0),(12,1),(12,5),(12,14),
        (13,2),(13,5),(13,6),(13,14),
        (14,3),(14,16),
        (15,6),(15,2),(15,4),(15,16),
        (16,7),(16,4),
        (17,0),(17,8),(17,11),(17,19),
        (18,8),(18,9),(18,6),(18,13),
        (19,10),(19,11),(19,16),
    ]
    edges.extend({"vertices":[a,b],"faces":[],"kind":"inside"} for a,b in links)
    return vertices, hull.equations, edges


VERTICES, FACE_PLANES, EDGES = make_geometry()


def rotation(phase):
    yaw = .14 + 2*math.pi*phase
    pitch = -.065 + .043*math.sin(2*math.pi*phase)
    roll = -.030 + .022*math.sin(2*math.pi*phase)
    cy,sy = math.cos(yaw),math.sin(yaw)
    cp,sp = math.cos(pitch),math.sin(pitch)
    cr,sr = math.cos(roll),math.sin(roll)
    return (np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]]) @
            np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]]) @
            np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]))


def controls(a, b, seed, amplitude=.017):
    tangent = b-a
    length = np.linalg.norm(tangent)
    tangent = tangent/length
    perpendicular = np.cross(tangent, [0.31,.72,.43])
    perpendicular /= np.linalg.norm(perpendicular)
    second = np.cross(tangent, perpendicular)
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-1,1,(7,2))*min(length*.012,amplitude)
    offsets[[0,-1]] = 0
    return (a+(b-a)*np.linspace(0,1,7)[:,None] +
            offsets[:,0,None]*perpendicular+offsets[:,1,None]*second)


EDGE_CONTROLS = [controls(*VERTICES[e["vertices"]], 431+i*101) for i,e in enumerate(EDGES)]


def make_hatches():
    # Engraving accents have fixed 3D coordinates, so they turn with the mark.
    groups = [
        ((-.42,1.69,.30),(1,0,0),[.18,.13,.10]),
        ((-1.50,.65,.08),(1,0,0),[.20,.27,.14,.11]),
        ((1.33,.89,.16),(1,0,0),[.12,.13,.13,.12]),
        ((.96,-.88,.33),(1,0,0),[.25,.20,.30]),
        ((-.93,-.86,.27),(1,0,0),[.14,.23,.18]),
        ((.06,-1.91,.22),(1,0,0),[.13,.20,.11]),
        ((-.35,1.49,-1.11),(1,0,0),[.17,.12,.08]),
        ((.58,-.94,-1.01),(1,0,0),[.23,.17,.11]),
        ((-1.14,.11,-.93),(0,0,1),[.14,.22]),
    ]
    result=[]
    for gi,(anchor,direction,lengths) in enumerate(groups):
        anchor=np.array(anchor,dtype=float)
        direction=np.array(direction,dtype=float)
        for j,length in enumerate(lengths):
            center=anchor+np.array([0,-j*.055,0])
            result.append(controls(center-direction*length*.5,center+direction*length*.5,798+gi*17+j,amplitude=.003))
    return result


HATCH_CONTROLS = make_hatches()


def project(points):
    perspective = 9.5/(9.5-points[:,2])
    return np.stack([384+139*points[:,0]*perspective,
                     384-139*points[:,1]*perspective],axis=1)


def ease(a,b,value):
    t=np.clip((value-a)/(b-a),0,1)
    return t*t*(3-2*t)


def frame_geometry(phase):
    matrix=rotation(phase)
    points=VERTICES@matrix.T
    normals=FACE_PLANES[:,:3]@matrix.T
    centers=(-FACE_PLANES[:,3,None]*FACE_PLANES[:,:3])@matrix.T
    directions=np.array([0,0,9.5])-centers
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    dots=np.sum(normals*directions,axis=1)
    lines=[]
    for index,edge in enumerate(EDGES):
        depth=float(np.mean(points[edge["vertices"],2]))
        near=float(ease(-1.15,1.15,depth))
        if edge["kind"]=="scaffold":
            front=float(ease(-.09,.13,max(dots[edge["faces"]])))
            opacity=.52+.45*front
            width=.90+.87*front
        else:
            opacity=.64+.30*near
            width=1.02+.24*near
        width*=1+.055*math.sin(index*3.71)
        lines.append((project(EDGE_CONTROLS[index]@matrix.T),opacity,width))
    for index,control in enumerate(HATCH_CONTROLS):
        transformed=control@matrix.T
        near=float(ease(-1.3,1.3,np.mean(transformed[:,2])))
        lines.append((project(transformed),.51+.35*near,1.20))
    projected=project(points)
    nodes=[]
    for i,point in enumerate(points):
        near=float(ease(-1.15,1.15,point[2]))
        radius=(4.1 if i in [0,2,3,4] else 3.3 if i<12 else 3.2)
        radius*=9.5/(9.5-point[2])
        nodes.append((projected[i],radius,.72+.26*near))
    return {"lines":lines,"nodes":nodes}


def path_d(points):
    p=[f"{x:.3f},{y:.3f}" for x,y in points]
    return f"M{p[0]} C{p[1]} {p[2]} {p[3]} C{p[4]} {p[5]} {p[6]}"


def line_halo(index, opacity, width, scale=HALO_SCALE):
    if index<len(EDGES) and EDGES[index]["kind"]=="scaffold":
        front=float(np.clip((opacity-.52)/.45,0,1))
        return (.72+.36*front)*scale, .76+.22*front
    if index<len(EDGES):
        near=float(np.clip((opacity-.64)/.30,0,1))
        return .84*scale, .82+.10*near
    return .75*scale,.82


def node_halo(opacity,scale=HALO_SCALE):
    near=float(np.clip((opacity-.72)/.26,0,1))
    return .95*scale,.87+.11*near


def render(phase,size=SIZE,transparent=True,halo_scale=HALO_SCALE):
    ss=2
    factor=size/SIZE*ss
    renderer=RendererAgg(size*ss,size*ss,72)
    renderer.clear()
    if not transparent:
        bg=MplPath([[0,0],[size*ss,0],[size*ss,size*ss],[0,size*ss],[0,0]],[1,2,2,2,79])
        gc=renderer.new_gc();gc.set_linewidth(0)
        renderer.draw_path(gc,bg,Affine2D(),(1,1,1,1))
    transform=Affine2D().scale(factor,-factor).translate(0,size*ss)
    frame=frame_geometry(phase)
    circle=MplPath.unit_circle()
    # Draw every white under-stroke/rim first. All ink follows globally, so
    # halos never erase other black strokes at network intersections.
    for halo_pass in [True,False]:
        for index,(points,opacity,width) in enumerate(frame["lines"]):
            if halo_pass:
                margin,opacity=line_halo(index,opacity,width,halo_scale)
                width+=2*margin
                color=(1,1,1,opacity)
            else:color=tuple(c/255 for c in INK)+(opacity,)
            gc=renderer.new_gc()
            gc.set_linewidth(width*factor)
            gc.set_capstyle("round");gc.set_joinstyle("round")
            gc.set_foreground(color);gc.set_antialiased(True)
            renderer.draw_path(gc,MplPath(points,[1,4,4,4,4,4,4]),transform)
        for center,radius,opacity in frame["nodes"]:
            if halo_pass:
                margin,opacity=node_halo(opacity,halo_scale)
                rx,ry=radius*1.02+margin,radius*.98+margin
                color=(1,1,1,opacity)
            else:
                rx,ry=radius*1.02,radius*.98
                color=tuple(c/255 for c in INK)+(opacity,)
            gc=renderer.new_gc();gc.set_linewidth(0)
            node_transform=Affine2D().scale(rx,ry).translate(*center)+transform
            renderer.draw_path(gc,circle,node_transform,color)
    image=Image.fromarray(np.array(renderer.buffer_rgba()))
    return image.resize((size,size),Image.Resampling.LANCZOS)


def write_svg(path,animated=False):
    ns="http://www.w3.org/2000/svg"
    ET.register_namespace("",ns)
    root=ET.Element("svg",xmlns=ns,width=str(SIZE),height=str(SIZE),viewBox="0 0 768 768",role="img",**{"aria-labelledby":"title desc"})
    ET.SubElement(root,"title",id="title").text="Ink crystal with a crisp white halo"
    ET.SubElement(root,"desc",id="desc").text="A transparent asymmetric crystal with fine dark ink, a narrow white halo, junction dots and engraving accents, turning in a seamless ten-second three-dimensional loop." if animated else "A transparent asymmetric crystal drawn in fine dark ink with a narrow white halo, junction dots and engraving accents."
    first=frame_geometry(0)
    frames=[frame_geometry(i/250) for i in range(251)] if animated else []
    def animate(item,name,values):
        ET.SubElement(item,"animate",attributeName=name,values=";".join(values),dur="10s",repeatCount="indefinite",calcMode="linear")
    for halo_pass in [True,False]:
        group=ET.SubElement(root,"g",id="white-halo" if halo_pass else "ink")
        color="#ffffff" if halo_pass else "#1d1f1e"
        for index,(points,opacity,width) in enumerate(first["lines"]):
            if halo_pass:
                margin,opacity=line_halo(index,opacity,width)
                width+=2*margin
            item=ET.SubElement(group,"path",d=path_d(points),fill="none",stroke=color,**{"stroke-width":f"{width:.3f}","stroke-linecap":"round","stroke-linejoin":"round","opacity":f"{opacity:.4f}"})
            if animated:
                animate(item,"d",[path_d(f["lines"][index][0]) for f in frames])
                if halo_pass:
                    animate(item,"opacity",[f'{line_halo(index,f["lines"][index][1],f["lines"][index][2])[1]:.4f}' for f in frames])
                    animate(item,"stroke-width",[f'{f["lines"][index][2]+2*line_halo(index,f["lines"][index][1],f["lines"][index][2])[0]:.3f}' for f in frames])
                else:
                    animate(item,"opacity",[f'{f["lines"][index][1]:.4f}' for f in frames])
                    animate(item,"stroke-width",[f'{f["lines"][index][2]:.3f}' for f in frames])
        for index,(center,radius,opacity) in enumerate(first["nodes"]):
            margin=0.
            if halo_pass:margin,opacity=node_halo(opacity)
            item=ET.SubElement(group,"ellipse",cx=f"{center[0]:.3f}",cy=f"{center[1]:.3f}",rx=f"{radius*1.02+margin:.3f}",ry=f"{radius*.98+margin:.3f}",fill=color,opacity=f"{opacity:.4f}")
            if animated:
                animate(item,"cx",[f'{f["nodes"][index][0][0]:.3f}' for f in frames])
                animate(item,"cy",[f'{f["nodes"][index][0][1]:.3f}' for f in frames])
                animate(item,"rx",[f'{f["nodes"][index][1]*1.02+(node_halo(f["nodes"][index][2])[0] if halo_pass else 0):.3f}' for f in frames])
                animate(item,"ry",[f'{f["nodes"][index][1]*.98+(node_halo(f["nodes"][index][2])[0] if halo_pass else 0):.3f}' for f in frames])
                animate(item,"opacity",[f'{node_halo(f["nodes"][index][2])[1] if halo_pass else f["nodes"][index][2]:.4f}' for f in frames])
    write_atomic(path,ET.tostring(root,encoding="utf-8",xml_declaration=True))


def samples():
    OUT.mkdir(exist_ok=True)
    qa=ROOT/"qa";qa.mkdir(exist_ok=True)
    backgrounds=[("white","#ffffff"),("black","#000000"),("gray","#777777")]
    comparison=Image.new("RGB",(1152,876),"#e8e8e8")
    draw=ImageDraw.Draw(comparison)
    for row,(label,amount) in enumerate([("STANDARD CRISP HALO",1.0),("FINER HALO",.72)]):
        picture=render(0,768,True,halo_scale=amount)
        for col,(name,color) in enumerate(backgrounds):
            composite=Image.alpha_composite(Image.new("RGBA",picture.size,color),picture).convert("RGB")
            save_image_atomic(composite,qa/f"halo-{label.split()[0].lower()}-{name}-768.png")
            comparison.paste(composite.resize((384,384),Image.Resampling.LANCZOS),(col*384,row*438+34))
            draw.text((col*384+14,row*438+13),f"{label} / {name.upper()}",fill="#202222")
    save_image_atomic(comparison,qa/"halo-comparison.png")
    small=Image.new("RGB",(768,288),"#e8e8e8")
    draw=ImageDraw.Draw(small)
    picture=render(0,256,True)
    for col,(name,color) in enumerate(backgrounds):
        composite=Image.alpha_composite(Image.new("RGBA",picture.size,color),picture).convert("RGB")
        small.paste(composite,(col*256,32))
        draw.text((col*256+12,12),f"256 PX / {name.upper()}",fill="#202222")
    save_image_atomic(small,qa/"halo-comparison-small.png")
    save_image_atomic(render(0,1536,True),OUT/"crystal-detailed-static.png")
    write_svg(OUT/"crystal-detailed-static.svg")
    data={"description":"Same detailed crystal network, with a crisp white under-stroke and node rim on a transparent canvas.","duration_seconds":10,"frame_rate":25,"canvas":[768,768],"halo_scale":HALO_SCALE,"vertices":VERTICES.tolist(),"edges":EDGES,"face_planes":FACE_PLANES.tolist(),"hatch_controls":[p.tolist() for p in HATCH_CONTROLS]}
    write_atomic(ROOT/"crystal-halo-geometry.json",(json.dumps(data,indent=2)+"\n").encode())
    print(f"Samples ready: {qa}",flush=True)


def full(resume=False):
    samples()
    frames_dir=ROOT/"frames-rgba"
    frames_dir.mkdir(exist_ok=True)
    encoder=_webp.WebPAnimEncoder((SIZE,SIZE),0,0,False,9,17,False,False)
    for index in range(250):
        target=frames_dir/f"frame-{index:04d}.png"
        frame=None
        if resume and target.exists():
            try:
                with Image.open(target) as previous:
                    previous.load()
                    if previous.mode=="RGBA" and previous.size==(SIZE,SIZE):
                        frame=previous.copy()
            except (OSError,SyntaxError,ValueError):pass
        if frame is None:
            frame=render(index/250,transparent=True)
            save_image_atomic(frame,target)
        encoder.add(frame.getim(),index*40,True,WEBP_EFFORT,100,WEBP_METHOD)
        if index%25==0:print(f"Master frame {index}/250",flush=True)
    encoder.add(None,10000,True,WEBP_EFFORT,100,0)
    webp=encoder.assemble("","","")
    if not webp:raise RuntimeError("WebP encoder produced no bytes")
    write_atomic(OUT/"crystal-detailed-loop.webp",webp)
    write_svg(OUT/"crystal-detailed-loop.svg",animated=True)
    print(f"All exports complete: {OUT}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--samples",action="store_true")
    parser.add_argument("--full",action="store_true")
    parser.add_argument("--resume",action="store_true",help="Reuse finalized frames only when render settings are unchanged")
    args=parser.parse_args()
    full(resume=args.resume) if args.full else samples()
